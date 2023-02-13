# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: MIT License

from __future__ import annotations
from argparse import ArgumentParser
from pathlib import Path
import os

import pandas as pd
import numpy as np
import lmdb
from pymatgen.core import Molecule
from tqdm import tqdm

from ocpmodels.datasets.generate_subsplit import write_data

"""
Script for converting OE62 formatted JSON data into a consistent LMDB
format within the Open MatSci ML Toolkit.

Arguments to this script are `-i` and `-o`, which are the input and output
paths respectively. For the former, target the JSON file containing the OE62
data, and the latter should be a directory that will hold the LMDB file.
"""

parser = ArgumentParser()
parser.add_argument("-i", "--input", type=Path, help="Path to the JSON file containing OE62 data.")
parser.add_argument("-o", "--output", type=Path, help="Path to directory to save the LMDB output.")

args = parser.parse_args()

input_file = Path(args.input)
assert input_file.exists(), f"{input_file} does not exist!"

os.makedirs(args.output, exist_ok=True)

df = pd.read_json(str(input_file), orient="split")

target_path = Path(args.output)

with lmdb.open(str(target_path.joinpath("oe62.lmdb")), subdir=False, map_size=int(1e9), meminit=False, map_async=True) as target_file:
    for index, row in tqdm(df.iterrows(), desc="OE62 rows processed.", total=len(df)):
        row_data = row.to_dict()
        row_data["homo-lumo_gap"] = row_data["energies_unocc_pbe"][0] - row_data["energies_occ_pbe"][-1]
        # convert the XYZ coordinates into Molecule object
        molecule = Molecule.from_str(row_data["xyz_pbe_relaxed"], fmt="xyz").get_centered_molecule()
        row_data["atomic_numbers"] = molecule.atomic_numbers
        # saves a NumPy array of cartesian coordinates
        row_data["pos"] = molecule.cart_coords.astype(np.float32)
        row_data["distance_matrix"] = molecule.distance_matrix.astype(np.float32)
        # now delete some keys
        good_keys = ["homo-lumo_gap", "atomic_numbers", "pos", "distance_matrix", "canonical_smiles", "refcode_csd"]
        for key in list(row_data.keys()):
            if key not in good_keys:
                del row_data[key]
        write_data(index, row_data, target_file)
