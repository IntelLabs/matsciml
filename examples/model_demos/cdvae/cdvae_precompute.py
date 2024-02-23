from typing import Dict, Union
from argparse import ArgumentParser, Namespace
from pathlib import Path
import os, sys, shutil, warnings, pickle
import lmdb, dgl, torch, numpy, time
from tqdm import tqdm
from torch_geometric.data import Data, Batch
from enum import Enum

from pymatgen.analysis.graphs import StructureGraph
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure
from matsciml.datasets.utils import atomic_number_map
from pymatgen.analysis import local_env

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append("{}/../".format(dir_path))

from matsciml.datasets.utils import connect_db_read, write_lmdb_data

MAX_ATOMS = 25

CrystalNN = local_env.CrystalNN(
    distance_cutoffs=None, x_diff_weight=-1, porous_adjustment=False
)  # , search_cutoff=15.0)

Atomic_num_map_global = atomic_number_map()

def get_atomic_num(elements):
    return [Atomic_num_map_global[element] for element in elements]

def get_atoms_from_atomic_numbers(atomic_numbers):
    map_reversed = {atomic_num: element for element, atomic_num in Atomic_num_map_global.items()}
    return [map_reversed[atomic_num] for atomic_num in atomic_numbers]

def get_distance_matrix(coords, lattice_vectors):
    # Create a 3D array containing all pairwise differences
    delta = coords[:, numpy.newaxis, :] - coords[numpy.newaxis, :, :]
    # Calculate the distances using vectorized operations
    distance_matrix = numpy.linalg.norm(numpy.dot(delta, lattice_vectors), axis=2)
    return distance_matrix

def get_jimage(structure):
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
    return to_jimages, edge_indices

def get_lattice(lattice):
    lattice_params = torch.FloatTensor(lattice.abc + tuple(lattice.angles))
    return {"lattice_params": lattice_params}

def processing_data(structure, return_dict, y):
    """
    The same as OG with the addition of jimages field
    """
    to_jimages, edge_indices = get_jimage(structure)
    return_dict["to_jimages"] = torch.LongTensor(to_jimages)
    return_dict["edge_index"] = torch.LongTensor(edge_indices).T
    edge_index = return_dict["edge_index"]  # torch.LongTensor([[0, 1], [1, 0]])
    return_dict["lattice_features"] = get_lattice(structure.lattice)
    lattice_params = return_dict["lattice_features"]["lattice_params"]
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

def parse_structure_MP(item, structure) -> None:
    return_dict = {}
    coords = torch.from_numpy(structure.cart_coords).float()
    return_dict["pos"] = coords[None, :] - coords[:, None]
    return_dict["coords"] = coords
    return_dict["frac_coords"] = structure.frac_coords
    atom_numbers = torch.LongTensor(structure.atomic_numbers)
    return_dict["atomic_numbers"] = torch.LongTensor(structure.atomic_numbers)     # keep atomic numbers for graph featurization
    return_dict["num_particles"] = len(atom_numbers)
    return_dict["distance_matrix"] = torch.from_numpy(structure.distance_matrix).float()
    y = item.get("formation_energy_per_atom") or 1
    data = processing_data(structure, return_dict, y)
    return data

def parse_structure_NOMAD(item, structure) -> None:
    return_dict = {}
    cartesian_coords = numpy.array(structure["cartesian_site_positions"])
    lattice_vectors = Lattice(numpy.array(structure["lattice_vectors"]))
    species = structure["species_at_sites"]
    frac_coords = lattice_vectors.get_fractional_coords(cart_coords=cartesian_coords)
    coords = torch.from_numpy(cartesian_coords)
    return_dict["pos"] = coords[None, :] - coords[:, None]
    return_dict["coords"] = coords
    return_dict["frac_coords"] = frac_coords
    num_particles = len(species)
    atom_numbers = get_atomic_num(species)
    return_dict["atomic_numbers"] = torch.LongTensor(atom_numbers)     # keep atomic numbers for graph featurization
    return_dict["num_particles"] = num_particles
    distance_matrix = get_distance_matrix(cartesian_coords, numpy.array(structure["lattice_vectors"]))
    return_dict["distance_matrix"] = torch.from_numpy(distance_matrix)
    y = (item["energies"]["total"]["value"]) / num_particles   #formation_energy_per_atom, eV
    structure = Structure(lattice_vectors, species, frac_coords)
    data = processing_data(structure, return_dict, y)
    return data

def parse_structure_OQMD(item, structure) -> None:
    return_dict = {}
    cartesian_coords = numpy.array(structure)
    lattice_vectors = Lattice(numpy.array(item["unit_cell"]))
    species = [site.split(" ")[0] for site in item["sites"]]
    frac_coords = lattice_vectors.get_fractional_coords(cart_coords=cartesian_coords)
    coords = torch.from_numpy(cartesian_coords)
    return_dict["pos"] = coords[None, :] - coords[:, None]
    return_dict["coords"] = coords
    return_dict["frac_coords"] = frac_coords
    return_dict["atomic_numbers"] = torch.LongTensor(item["atomic_numbers"])     # keep atomic numbers for graph featurization
    return_dict["num_particles"] = item["natoms"]
    distance_matrix = get_distance_matrix(cartesian_coords, numpy.array(item["unit_cell"]))
    return_dict["distance_matrix"] = torch.from_numpy(distance_matrix).float()
    y = item["delta_e"]
    structure = Structure(lattice_vectors, species, frac_coords)
    data = processing_data(structure, return_dict, y)
    return data

def parse_structure_Carolina(item, structure) -> None:
    return_dict = {}
    cartesian_coords = structure
    a, b, c, alpha, beta, gamma = [
                    float(item["_cell_length_a"]),
                    float(item["_cell_length_b"]),
                    float(item["_cell_length_c"]),
                    float(item["_cell_angle_alpha"]),
                    float(item["_cell_angle_beta"]),
                    float(item["_cell_angle_gamma"])]
    lattice_vectors = Lattice.from_parameters(a, b, c, alpha, beta, gamma)
    species = get_atoms_from_atomic_numbers(item["atomic_numbers"])
    frac_coords = lattice_vectors.get_fractional_coords(cart_coords=cartesian_coords)
    coords = torch.from_numpy(cartesian_coords)
    return_dict["pos"] = coords[None, :] - coords[:, None]
    return_dict["coords"] = coords
    return_dict["frac_coords"] = frac_coords
    atom_numbers = torch.LongTensor(item["atomic_numbers"])
    return_dict["atomic_numbers"] = atom_numbers     # keep atomic numbers for graph featurization
    return_dict["num_particles"] = len(item["atomic_numbers"])
    distance_matrix = get_distance_matrix(cartesian_coords, lattice_vectors._matrix)
    return_dict["distance_matrix"] = torch.from_numpy(distance_matrix)
    y = item["energy"]
    structure = Structure(lattice_vectors, species, frac_coords)
    data = processing_data(structure, return_dict, y)
    return data

def check_num_atoms(num_atoms):
    if num_atoms is not None:
        if num_atoms > MAX_ATOMS:
            print("Number of atoms is larger than MAX_ATOMS.")
            return False
        return True
    else:
        warnings.warn("One entry is skipped due to missing the number of atoms, which is needed by the workflow!")

def check_structure(structure):
    if structure is None:
        raise ValueError("Structure not found in data - workflow needs a structure to use!")
    else:
        return True

def data_to_cdvae(item, dataset_name):
    """
    Convert data to PyTorch Geometric format for a given dataset.

    Args:
    - item: The data item to be converted.
    - dataset_name: Name of the dataset.

    Returns:
    - PyTorch Geometric data object.
    """
    if dataset_name == "MP":
        num_atoms = len(item["structure"].atomic_numbers)
        structure = item.get("structure", None)
        parse_structure_func = parse_structure_MP
    elif dataset_name == "NOMAD":
        num_atoms = len(item["properties"]["structures"]["structure_conventional"]["species_at_sites"])
        structure = item["properties"]["structures"].get("structure_conventional", None)
        parse_structure_func = parse_structure_NOMAD
    elif dataset_name == "OQMD":
        num_atoms = item["natoms"]
        structure = item.get("cart_coords", None)
        parse_structure_func = parse_structure_OQMD
    elif dataset_name == "Carolina":
        num_atoms = len(item["atomic_numbers"])
        structure = item.get("cart_coords", None)
        parse_structure_func = parse_structure_Carolina
    else:
        raise ValueError("Invalid dataset name provided.")

    check_num_atoms(num_atoms)
    check_structure(structure)
    if check_num_atoms and check_structure:
        pyg_data = parse_structure_func(item, structure)
        return pyg_data
    else:
        return None


#############################################################################
def main(args: Namespace):
    start_time = time.time()
    print("Start")
    input_path = Path(args.src_lmdb)
    output_path = Path(args.output_folder)
    if not input_path.exists():
        raise FileNotFoundError(f"{input_path} could not be found.")
    if args.dataset is None:
        raise FileNotFoundError("Dataset name was not provided.")
    if output_path.exists():
        raise ValueError(f"{output_path} already exists, please check its contents and remove the folder!")
    os.makedirs(output_path, exist_ok=True)
    db_paths = sorted(input_path.glob("*.lmdb"))

    # loop over individual LMDB files within the input set, if there are more than one
    for path in db_paths:
        target = output_path / path.name
        pyg_env = connect_db_read(path)
        # open the output file for writing
        target_env = lmdb.open(
            str(target),
            subdir=False,
            # The map_size setup is different for Windows vs. Linux:
            # https://github.com/tensorpack/tensorpack/issues/1209
            # https://stackoverflow.com/questions/33508305/lmdb-maximum-size-of-the-database-for-windows
            map_size=1099511627776 * 2, ## 1099511627776 = 1TB, 1073741824 = 1 GB, 104857600 = 100 MB, 1048576 = 1 MB
            meminit=False,
            map_async=True,
        )
        with pyg_env.begin() as txn:
            print(f"There are {txn.stat()['entries']} entries.")
            keys = [key for key in txn.cursor().iternext(values=False)]
            for i, key in enumerate(tqdm(keys)):
                dataset_name = args.dataset
                if key.decode("utf-8").isdigit():
                    crystal_data = pickle.loads(txn.get(key))
                    pyg_data = data_to_cdvae(crystal_data, dataset_name)
                    if pyg_data is not None:
                        key = key.decode("utf-8")
                        write_lmdb_data(key, pyg_data, target_env)
                else:
                    metadata = pickle.loads(txn.get(key))
                    key = key.decode("utf-8")
                    write_lmdb_data(key, metadata, target_env)

    end_time = time.time()
    running_time = end_time - start_time
    minutes, seconds = divmod(running_time, 60)
    print(f"Done! Program executed in {int(minutes)} minutes and {int(seconds)} seconds")


if __name__ == "__main__":
    parser = ArgumentParser()
    dataset = ["MP", "NOMAD", "OQMD", "Carolina"]
    parser.add_argument("--src_lmdb", "-i", type=str, help="Folder containing the source LMDB files to be converted.")
    parser.add_argument("--dataset", "-d", type=str, choices=dataset, help="Select one of the datasets.")
    parser.add_argument("--output_folder", "-o", type=str, help="Path to a non-existing folder to save processed data to.")
    # CLI example: python examples/model_demos/cdvae/cdvae_precompute.py -i matsciml/datasets/materials_project/devset -d MP -o matsciml/datasets/materials_project/devset/cdvae
    args = parser.parse_args()
    print(args)
    main(args)