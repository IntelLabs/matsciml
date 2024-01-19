from __future__ import annotations

import os
import pickle
import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict, Union

import dgl
import lmdb
import torch
from pymatgen.analysis import local_env
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.core import Structure
from torch_geometric.data import Batch, Data
from tqdm import tqdm

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f"{dir_path}/../")

from matsciml.datasets.generate_subsplit import connect_db_read, write_data

MAX_ATOMS = 25

CrystalNN = local_env.CrystalNN(
    distance_cutoffs=None,
    x_diff_weight=-1,
    porous_adjustment=False,
)  # , search_cutoff=15.0)


def parse_structure(item) -> None:
    """
    The same as OG with the addition of jimages field
    """
    return_dict = {}
    structure = item.get("structure", None)
    if structure is None:
        raise ValueError(
            "Structure not found in data - workflow needs a structure to use!",
        )
    coords = torch.from_numpy(structure.cart_coords).float()
    return_dict["pos"] = coords[None, :] - coords[:, None]
    return_dict["coords"] = coords
    return_dict["frac_coords"] = structure.frac_coords
    atom_numbers = torch.LongTensor(structure.atomic_numbers)
    # keep atomic numbers for graph featurization
    return_dict["atomic_numbers"] = torch.LongTensor(structure.atomic_numbers)
    return_dict["num_particles"] = len(atom_numbers)
    return_dict["distance_matrix"] = torch.from_numpy(structure.distance_matrix).float()
    # jimages

    #
    try:
        crystal_graph = StructureGraph.with_local_env_strategy(structure, CrystalNN)
    except ValueError:
        return None

    edge_indices, to_jimages = [], []
    for i, j, to_jimage in crystal_graph.graph.edges(data="to_jimage"):
        edge_indices.append([j, i])
        to_jimages.append(to_jimage)
        edge_indices.append([i, j])
        to_jimages.append(tuple(-tj for tj in to_jimage))
    return_dict["to_jimages"] = torch.LongTensor(to_jimages)
    return_dict["edge_index"] = torch.LongTensor(edge_indices).T

    # grab lattice properties
    lattice_params = torch.FloatTensor(
        structure.lattice.abc + tuple(structure.lattice.angles),
    )
    lattice_features = {
        "lattice_params": lattice_params,
    }
    return_dict["lattice_features"] = lattice_features

    edge_index = return_dict["edge_index"]  # torch.LongTensor([[0, 1], [1, 0]])
    lattice_params = return_dict["lattice_features"]["lattice_params"]
    y = item["formation_energy_per_atom"]
    prop = torch.Tensor([y])

    # atom_coords are fractional coordinates
    # edge_index is incremented during batching
    # https://pytorch-geometric.readthedocs.io/en/latest/notes/batching.html
    data = Data(
        frac_coords=torch.Tensor(return_dict["frac_coords"]),
        atom_types=torch.LongTensor(return_dict["atomic_numbers"]),
        lengths=torch.Tensor(lattice_params[:3]).view(1, -1),
        angles=torch.Tensor(lattice_params[3:]).view(1, -1),
        edge_index=edge_index,  # shape (2, num_edges)
        to_jimages=return_dict["to_jimages"],
        num_atoms=len(return_dict["atomic_numbers"]),
        num_bonds=edge_index.shape[1],
        num_nodes=len(
            return_dict["atomic_numbers"],
        ),  # special attribute used for batching in pytorch geometric
        y=prop.view(1, -1),
    )
    return data


def convert_pyg_to_dgl(pyg_graph) -> dict[str, dgl.DGLGraph | torch.Tensor]:
    # bijective mapping from PyG to DGL
    (u, v) = pyg_graph.edge_index
    dgl_graph = dgl.graph((u, v), num_nodes=pyg_graph.natoms)
    # loop over the node and edge attributes, and copy the data over
    for node_key in ["atomic_numbers", "force", "pos", "pos_relaxed", "fixed", "tags"]:
        dgl_graph.ndata[node_key] = getattr(pyg_graph, node_key)
    for edge_key in ["cell_offsets", "distances"]:
        dgl_graph.edata[edge_key] = getattr(pyg_graph, edge_key)
    return_data = {"graph": dgl_graph}
    for key in ["cell", "sid", "y_init", "y_relaxed", "natoms"]:
        return_data[key] = getattr(pyg_graph, key)
    return return_data


def data_to_cdvae(item):
    num_atoms = len(item["structure"].atomic_numbers)
    if num_atoms > MAX_ATOMS:
        return None

    pyg_data = parse_structure(item)
    return pyg_data


def main(args: Namespace):
    print("Start")
    input_path = Path(args.src_lmdb)
    output_path = Path(args.output_folder)
    if not input_path.exists():
        raise FileNotFoundError(f"{input_path} could not be found.")
    if output_path.exists():
        raise ValueError(
            f"{output_path} already exists, please check its contents and remove the folder!",
        )
    os.makedirs(output_path)
    db_paths = sorted(input_path.glob("*.lmdb"))
    # loop over individual LMDB files within the input set, if there are
    # more than one
    for path in db_paths:
        target = output_path / path.name
        pyg_env = connect_db_read(path)
        # grab the keys
        with pyg_env.begin() as txn:
            keys = [key for key in txn.cursor().iternext(values=False)]
        # open the output file for writing
        target_env = lmdb.open(
            str(target),
            subdir=False,
            map_size=1099511627776 * 2,
            meminit=False,
            map_async=True,
        )
        with pyg_env.begin() as txn:
            for i, key in enumerate(tqdm(keys)):
                # for digit encodings, these are indexes of data points
                # that we want to convert
                if key.decode("utf-8").isdigit():
                    crystal_data = pickle.loads(txn.get(key))
                    pyg_data = data_to_cdvae(crystal_data)
                    if pyg_data is not None:
                        # convert the key before writing
                        key = key.decode("utf-8")
                        write_data(key, pyg_data, target_env)
                # otherwise it's just metadata to copy over directly
                else:
                    metadata = pickle.loads(txn.get(key))
                    # convert the key before writing
                    key = key.decode("utf-8")
                    write_data(key, metadata, target_env)

    print("Done")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--src_lmdb",
        "-i",
        type=str,
        help="Folder containing the source LMDB files to be converted.",
    )
    parser.add_argument(
        "--output_folder",
        "-o",
        type=str,
        help="Path to a non-existing folder to save processed data to.",
    )
    args = parser.parse_args()
    main(args)
