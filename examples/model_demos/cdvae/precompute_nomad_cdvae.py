from typing import Dict, Union
from argparse import ArgumentParser, Namespace
from pathlib import Path
import os, sys, shutil, warnings, pickle
import lmdb, dgl, torch, numpy, time
from tqdm import tqdm
from torch_geometric.data import Data, Batch

from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.analysis import local_env
from matsciml.datasets.utils import atomic_number_map

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append("{}/../".format(dir_path))

from matsciml.datasets.utils import connect_db_read, write_lmdb_data

MAX_ATOMS = 25

CrystalNN = local_env.CrystalNN(
    distance_cutoffs=None, x_diff_weight=-1, porous_adjustment=False
)  # , search_cutoff=15.0)

def get_atomic_num(elements):
    atomic_num_dict = []
    for element in elements:
        atomic_num = atomic_number_map()[element]
        if atomic_num is not None:
            atomic_num_dict.append(atomic_num)
    return atomic_num_dict

def get_distance_matrix(coords, lattice_vectors):
    num_sites = len(coords)
    distance_matrix = numpy.zeros((num_sites, num_sites))
    for i in range(num_sites):
        for j in range(i, num_sites):
            delta = numpy.subtract(coords[i], coords[j])
            # Apply minimum image convention for periodic boundary conditions
            # delta -= numpy.round(delta)
            distance = numpy.linalg.norm(numpy.dot(delta, lattice_vectors))
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance
    return distance_matrix

def parse_structure(item) -> None:
    """
    The same as OG with the addition of jimages field
    """
    return_dict = {}
    structure = item["properties"]["structures"].get("structure_conventional", None)
    if structure is None:
        raise ValueError(
            "Structure not found in data - workflow needs a structure to use!"
        )
    cartesian_coords = numpy.array(structure["cartesian_site_positions"]) * 1E10
    lattice_vectors = Lattice(numpy.array(structure["lattice_vectors"]) * 1E10)
    species = structure["species_at_sites"]
    frac_coords = lattice_vectors.get_fractional_coords(cart_coords=cartesian_coords)
    coords = torch.from_numpy(cartesian_coords)
    return_dict["pos"] = coords[None, :] - coords[:, None]
    return_dict["coords"] = coords
    return_dict["frac_coords"] = frac_coords
    num_particles = len(species)
    atom_numbers = get_atomic_num(species)
    # keep atomic numbers for graph featurization
    return_dict["atomic_numbers"] = torch.LongTensor(atom_numbers)
    return_dict["num_particles"] = num_particles
    distance_matrix = get_distance_matrix(cartesian_coords, numpy.array(structure["lattice_vectors"]) * 1E10)
    return_dict["distance_matrix"] = torch.from_numpy(distance_matrix)
    # jimages
    structure_new = Structure(lattice_vectors, species, frac_coords)
    #
    try:
        crystal_graph = StructureGraph.with_local_env_strategy(structure_new, CrystalNN)
    except ValueError:
        return None
    # print(crystal_graph, "\n")
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
        lattice_vectors.abc + tuple(lattice_vectors.angles)
    )
    lattice_features = {
        "lattice_params": lattice_params,
    }
    return_dict["lattice_features"] = lattice_features

    edge_index = return_dict["edge_index"]  # torch.LongTensor([[0, 1], [1, 0]])
    lattice_params = return_dict["lattice_features"]["lattice_params"]
    y = (item["energies"]["total"]["value"] * 6.241509074461E+18) / num_particles   #formation_energy_per_atom, eV
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
        num_nodes=len(return_dict["atomic_numbers"]),  # special attribute used for batching in pytorch geometric
        y=prop.view(1, -1),
    )
    return data

def convert_pyg_to_dgl(pyg_graph) -> Dict[str, Union[dgl.DGLGraph, torch.Tensor]]:
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
    num_atoms = len(item["properties"]["structures"]["structure_conventional"]["species_at_sites"])
    # print(num_atoms)
    if num_atoms > MAX_ATOMS:
        return None 
    pyg_data = parse_structure(item)
    return pyg_data

def main(args: Namespace):
    start_time = time.time()
    print("Start")
    inumpyut_path = Path(args.src_lmdb)
    output_path = Path(args.output_folder)
    if not inumpyut_path.exists():
        raise FileNotFoundError(f"{inumpyut_path} could not be found.")
    if output_path.exists():
        raise ValueError(
            f"{output_path} already exists, please check its contents and remove the folder!"
        )
    # if output_path.exists():
    #     warnings.warn(
    #         f"{output_path} already exists, please check its contents and remove the folder!"
    #     )
    #     shutil.rmtree(output_path)

    os.makedirs(output_path, exist_ok=True)
    db_paths = sorted(inumpyut_path.glob("*.lmdb"))
    # loop over individual LMDB files within the inumpyut set, if there are
    # more than one
    for path in db_paths:
        target = output_path / path.name
        pyg_env = connect_db_read(path)
        # open the output file for writing
        target_env = lmdb.open(
            str(target),
            subdir=False,
            # For Linux, map_size can be set as high as possible such as 1099511627776 = 1TB, higher than the physical memory size,
            # and the final file size is reduced to the needed memory size. 
            # But for Windows system, it needs to be carefully determined. It cannot go higher than the physical memory size, and you don't want to set it too high as well.
            # because the final file size will be the same map_size no matter it is needed or not. Thus, 10737418240 = 10 GB is OK.
            # https://github.com/tensorpack/tensorpack/issues/1209
            # https://stackoverflow.com/questions/33508305/lmdb-maximum-size-of-the-database-for-windows
            map_size=1048576, # 1099511627776 = 1TB, 1073741824 = 1 GB, 104857600 = 100 MB, 1048576 = 1 MB
            meminit=False,
            map_async=True,
        )
        with pyg_env.begin() as txn:
            length = txn.stat()['entries']
            print(f"There are {length} entries.")
            for i in tqdm(range(length)):
                key = f'{i}'.encode('utf-8')
                crystal_data = pickle.loads(txn.get(key))
                pyg_data = data_to_cdvae(crystal_data)
                # print("pyg_data = ", pyg_data, "\n")
                if pyg_data is not None:
                    write_lmdb_data(str(i), pyg_data, target_env)
                # otherwise it's just metadata to copy over directly
                else:
                    write_lmdb_data(str(i), crystal_data, target_env)

    end_time = time.time()
    running_time = end_time - start_time
    minutes, seconds = divmod(running_time, 60)
    print(f"Done! Program executed in {int(minutes)} minutes and {int(seconds)} seconds")


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

        # Check if the command-line arguments were provided. If not, set default values or read from a configuration file.
    if args.src_lmdb is None:
        args.src_lmdb = "matsciml/datasets/nomad/devset"  # Set default value or read from config file
        # args.src_lmdb = "matsciml/datasets/oqmd/all"  # Set default value or read from config file
        # args.src_lmdb = "matsciml/datasets/carolina_matdb/all"  # Set default value or read from config file
        # args.src_lmdb = "matsciml/datasets/mp-project/base/test"  # Set default value or read from config file

    if args.output_folder is None:
        # args.output_folder = "matsciml/output/nomad"    # Set default value or read from config file
        args.output_folder = args.src_lmdb + "/cdvae"

    main(args)
