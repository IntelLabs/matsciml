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

def input_folder_path():
    while True:
        folder_path = input("Enter the path of the folder containing dataset (Example: matsciml/datasets/materials_project/devset or exit): ")
        if folder_path == "exit":
            break
        if os.path.exists(folder_path) and os.path.isdir(folder_path):
            return folder_path
        else:
            print("Invalid folder path. Please enter a valid directory path.")

def select_dataset():
    print("\nSelect a dataset type:")
    print("1. Dataset Materials Project")
    print("2. Dataset NOMAD")
    print("3. Dataset OQMD")
    print("4. Dataset Carolina\n")
    choice = input("Enter the number corresponding to your choice: ")

    if choice == '1':
        selected_dataset = "MP"
    elif choice == '2':
        selected_dataset = "NOMAD"
    elif choice == '3':
        selected_dataset = "OQMD"
    elif choice == '4':
        selected_dataset = "Carolina"
    else:
        raise ValueError(f"Invalid choice. Please enter a number between 1 and 4.")
    return selected_dataset

def output_folder_path():
    while True:
        folder_path = input("Enter the path of the output folder (Example: matsciml/datasets/materials_project/devset/output or exit): ")
        if folder_path == "exit":
            break
        if os.path.exists(folder_path) and os.path.isdir(folder_path):
            return folder_path
        else:
            print("Invalid folder path. Please enter a valid directory path.")


def get_atomic_num(elements):
    atomic_num_dict = []
    for element in elements:
        atomic_num = atomic_number_map()[element]
        if atomic_num is not None:
            atomic_num_dict.append(atomic_num)
    return atomic_num_dict

def get_atoms_from_atomic_numbers(atomic_numbers):
    map_reversed = {atomic_num: element for element, atomic_num in atomic_number_map().items()}
    # Create a list to store the elements
    elements = []
    # Iterate through the atomic numbers
    for atomic_num in atomic_numbers:
        element = map_reversed[atomic_num]
        if element is not None:
            elements.append(element)
    return elements

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

def processing_data(return_dict, lattice_params, edge_index, prop):
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

def parse_structure_MP(item) -> None:
    """
    The same as OG with the addition of jimages field
    """
    return_dict = {}
    structure = item.get("structure", None)
    if structure is None:
        raise ValueError(
            "Structure not found in data - workflow needs a structure to use!"
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
        structure.lattice.abc + tuple(structure.lattice.angles)
    )
    lattice_features = {
        "lattice_params": lattice_params,
    }
    return_dict["lattice_features"] = lattice_features

    edge_index = return_dict["edge_index"]  # torch.LongTensor([[0, 1], [1, 0]])
    lattice_params = return_dict["lattice_features"]["lattice_params"]
    y = item.get("formation_energy_per_atom")
    if y is None:
        y = 1
    prop = torch.Tensor([y])
    data = processing_data(return_dict, lattice_params, edge_index, prop)
    return data

def parse_structure_NOMAD(item) -> None:
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
    data = processing_data(return_dict, lattice_params, edge_index, prop)
    return data

def parse_structure_OQMD(item) -> None:
    """
    The same as OG with the addition of jimages field
    """
    return_dict = {}
    structure = item.get("cart_coords", None)
    if structure is None:
        raise ValueError(
            "Structure not found in data - workflow needs a structure to use!"
        )
    cartesian_coords = numpy.array(structure)
    lattice_vectors = Lattice(numpy.array(item["unit_cell"]))
    species = [site.split(" ")[0] for site in item["sites"]]
    frac_coords = lattice_vectors.get_fractional_coords(cart_coords=cartesian_coords)
    coords = torch.from_numpy(cartesian_coords)
    return_dict["pos"] = coords[None, :] - coords[:, None]
    return_dict["coords"] = coords
    return_dict["frac_coords"] = frac_coords
    # keep atomic numbers for graph featurization
    return_dict["atomic_numbers"] = torch.LongTensor(item["atomic_numbers"])
    return_dict["num_particles"] = item["natoms"]
    distance_matrix = get_distance_matrix(cartesian_coords, numpy.array(item["unit_cell"]))
    return_dict["distance_matrix"] = torch.from_numpy(distance_matrix).float()
    # jimages
    structure_new = Structure(lattice_vectors, species, frac_coords)

    #
    try:
        crystal_graph = StructureGraph.with_local_env_strategy(structure_new, CrystalNN)
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
        lattice_vectors.abc + tuple(lattice_vectors.angles)
    )
    lattice_features = {
        "lattice_params": lattice_params,
    }
    return_dict["lattice_features"] = lattice_features

    edge_index = return_dict["edge_index"]  # torch.LongTensor([[0, 1], [1, 0]])
    lattice_params = return_dict["lattice_features"]["lattice_params"]
    y = item["delta_e"]
    prop = torch.Tensor([y])
    data = processing_data(return_dict, lattice_params, edge_index, prop)
    return data

def parse_structure_Carolina(item) -> None:
    """
    The same as OG with the addition of jimages field
    """
    return_dict = {}
    structure = item.get("cart_coords", None)
    if structure is None:
        raise ValueError(
            "Structure not found in data - workflow needs a structure to use!"
        )
    # print(item["_cell_length_a"])
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
    # keep atomic numbers for graph featurization
    return_dict["atomic_numbers"] = atom_numbers
    return_dict["num_particles"] = len(item["atomic_numbers"])
    distance_matrix = get_distance_matrix(cartesian_coords, lattice_vectors._matrix)
    return_dict["distance_matrix"] = torch.from_numpy(distance_matrix)
    # jimages
    structure_new = Structure(lattice_vectors, species, frac_coords)

    #
    try:
        crystal_graph = StructureGraph.with_local_env_strategy(structure_new, CrystalNN)
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
        lattice_vectors.abc + tuple(lattice_vectors.angles)
    )
    lattice_features = {
        "lattice_params": lattice_params,
    }
    return_dict["lattice_features"] = lattice_features

    edge_index = return_dict["edge_index"]  # torch.LongTensor([[0, 1], [1, 0]])
    lattice_params = return_dict["lattice_features"]["lattice_params"]
    y = item["energy"]
    prop = torch.Tensor([y])
    data = processing_data(return_dict, lattice_params, edge_index, prop)
    return data


def check_num_atoms(num_atoms):
    if num_atoms is not None:
        if num_atoms > MAX_ATOMS:
            return None
        return True
    else:
        warnings.warn("One entry is skipped due to missing the number of atoms, which is needed by the workflow!")

def data_to_cdvae_MP(item):
    num_atoms = len(item["structure"].atomic_numbers)
    check_num_atoms(num_atoms)
    if check_num_atoms:
        pyg_data = parse_structure_MP(item)
        return pyg_data

def data_to_cdvae_NOMAD(item):
    num_atoms = len(item["properties"]["structures"]["structure_conventional"]["species_at_sites"])
    check_num_atoms(num_atoms)
    if check_num_atoms:
        pyg_data = parse_structure_NOMAD(item)
        return pyg_data

def data_to_cdvae_OQMD(item):
    num_atoms = item["natoms"]
    check_num_atoms(num_atoms)
    if check_num_atoms:
        pyg_data = parse_structure_OQMD(item)
        return pyg_data

def data_to_cdvae_Carolina(item):
    num_atoms = len(item["atomic_numbers"])
    check_num_atoms(num_atoms)
    if check_num_atoms:
        pyg_data = parse_structure_Carolina(item)
        return pyg_data

#############################################################################
def main(args: Namespace):
    start_time = time.time()
    print("Start")
    input_path = Path(args.src_lmdb)
    output_path = Path(args.output_folder)

    if not input_path.exists():
        raise FileNotFoundError(f"{input_path} could not be found.")
    if args.dataset is None:
        raise FileNotFoundError(" Dataset name was not provided. Existing ones include MP, NOMAD, OQMD, Carolina, which are case-sensitive.")

    # if output_path.exists():
    #     raise ValueError(
    #         f"{output_path} already exists, please check its contents and remove the folder!"
    #     )

    # if output_path.exists():
    #     warnings.warn(
    #         f"{output_path} already exists, please check its contents and remove the folder!"
    #     )
    #     shutil.rmtree(output_path)

    os.makedirs(output_path, exist_ok=True)
    db_paths = sorted(input_path.glob("*.lmdb"))
    # loop over individual LMDB files within the input set, if there are
    # more than one
    for path in db_paths:
        target = output_path / path.name
        pyg_env = connect_db_read(path)
        # open the output file for writing
        target_env = lmdb.open(
            str(target),
            subdir=False,
            # For Linux, the map_size can be set as high as possible such as 1099511627776(1TB), exceeding the physical memory size,
            # and the final file size will be reduced to the required memory size.
            # But for Windows system, map_size needs to be chosen carefully. It shouldn't exceed the physical memory size, and setting it too high should be avoided.
            # This is because the final file size will be the same as the map_size, whether it is needed or not.
            # Thus, a value like 10737418240 (10GB) is sufficient for a complete datasets including MP, Carolina, NOMAD, OQMD.
            # For the devset, 10 MB is sufficient.
            # https://github.com/tensorpack/tensorpack/issues/1209
            # https://stackoverflow.com/questions/33508305/lmdb-maximum-size-of-the-database-for-windows
            map_size=10485760, ## 1099511627776 = 1TB, 1073741824 = 1 GB, 104857600 = 100 MB, 1048576 = 1 MB
            meminit=False,
            map_async=True,
        )
        with pyg_env.begin() as txn:
            print(f"There are {txn.stat()['entries']} entries.")
            keys = [key for key in txn.cursor().iternext(values=False)]
            if args.dataset == "MP":
                for i, key in enumerate(tqdm(keys)):
                    if key.decode("utf-8").isdigit():
                        crystal_data = pickle.loads(txn.get(key))
                        pyg_data = data_to_cdvae_MP(crystal_data)
                        if pyg_data is not None:
                            # convert the key before writing
                            key = key.decode("utf-8")
                            write_lmdb_data(key, pyg_data, target_env)
                    # otherwise it's just metadata to copy over directly
                    else:
                        metadata = pickle.loads(txn.get(key))
                        # convert the key before writing
                        key = key.decode("utf-8")
                        write_lmdb_data(key, metadata, target_env)
            if args.dataset == "NOMAD":
                for i, key in enumerate(tqdm(keys)):
                    if key.decode("utf-8").isdigit():
                        crystal_data = pickle.loads(txn.get(key))
                        pyg_data = data_to_cdvae_NOMAD(crystal_data)
                        if pyg_data is not None:
                            # convert the key before writing
                            key = key.decode("utf-8")
                            write_lmdb_data(key, pyg_data, target_env)
                    # otherwise it's just metadata to copy over directly
                    else:
                        metadata = pickle.loads(txn.get(key))
                        # convert the key before writing
                        key = key.decode("utf-8")
                        write_lmdb_data(key, metadata, target_env)
            if args.dataset == "OQMD":
                for i, key in enumerate(tqdm(keys)):
                    if key.decode("utf-8").isdigit():
                        crystal_data = pickle.loads(txn.get(key))
                        pyg_data = data_to_cdvae_OQMD(crystal_data)
                        if pyg_data is not None:
                            # convert the key before writing
                            key = key.decode("utf-8")
                            write_lmdb_data(key, pyg_data, target_env)
                    # otherwise it's just metadata to copy over directly
                    else:
                        metadata = pickle.loads(txn.get(key))
                        # convert the key before writing
                        key = key.decode("utf-8")
                        write_lmdb_data(key, metadata, target_env)
            if args.dataset == "Carolina":
                for i, key in enumerate(tqdm(keys)):
                    if key.decode("utf-8").isdigit():
                        crystal_data = pickle.loads(txn.get(key))
                        pyg_data = data_to_cdvae_Carolina(crystal_data)
                        if pyg_data is not None:
                            # convert the key before writing
                            key = key.decode("utf-8")
                            write_lmdb_data(key, pyg_data, target_env)
                    # otherwise it's just metadata to copy over directly
                    else:
                        metadata = pickle.loads(txn.get(key))
                        # convert the key before writing
                        key = key.decode("utf-8")
                        write_lmdb_data(key, metadata, target_env)


    end_time = time.time()
    running_time = end_time - start_time
    minutes, seconds = divmod(running_time, 60)
    print(f"Done! Program executed in {int(minutes)} minutes and {int(seconds)} seconds")


if __name__ == "__main__":
    dataset = select_dataset()
    input_path = input_folder_path()
    output_path = output_folder_path()
    parser = ArgumentParser()
    parser.add_argument("--src_lmdb", "-i", default=input_path, type=str, help="Folder containing the source LMDB files to be converted.")
    parser.add_argument("--dataset", "-d", default=dataset, type=str, help="Different datasets including MP, Carolina, OQMD, NOMAD.")
    parser.add_argument("--output_folder", "-o", default=output_path, type=str, help="Path to a non-existing folder to save processed data to.")
    args = parser.parse_args()
    print(args)
    main(args)