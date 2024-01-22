from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import lmdb
import numpy as np
import torch
from ase import Atoms
from ase.cell import Cell
from ase.io import read
from tqdm import tqdm

from matsciml.datasets.utils import write_lmdb_data


@dataclass
class LiPSStructure:
    """
    Data structure for parsing the LiPS data from
    https://archive.materialscloud.org/record/2022.45

    The main form of interaction for this class is to load in an extended XYZ
    file, and dump it as an LMDB file consistent with other datasets
    in matsciml.
    """

    def __init__(self, *atoms: Atoms) -> None:
        self.atoms = atoms

    def __len__(self) -> int:
        return len(self.atoms)

    def __iter__(self):
        iterator = iter(self.atoms)
        yield next(iterator)

    def __getitem__(self, index: int) -> Atoms:
        return self.atoms[index]

    @staticmethod
    def entry_to_dict(struct: Atoms) -> dict[str, torch.Tensor | float]:
        result = {
            "pos": struct.get_positions(),
            "cell": struct.get_cell(),
            "atomic_numbers": struct.get_atomic_numbers(),
            "energy": struct.get_potential_energy(),
            "force": struct.get_forces(),
            "pbc": struct.get_pbc(),
        }
        keys = list(result.keys())
        for key in keys:
            data = result[key]
            # for iterable types, format them as torch tensors
            if isinstance(data, Iterable):
                if isinstance(data, np.ndarray):
                    data = torch.from_numpy(result[key])
                elif isinstance(data, Cell):
                    data = data.tolist()
                    data = torch.Tensor(data)
                else:
                    print(f"{key} is unknown data type: {data}")
                if torch.is_floating_point(data):
                    # cast to single precision
                    data = data.float()
                else:
                    # keep atomic numbers long
                    data = data.long()
            result[key] = data
        return result

    @classmethod
    def from_xyz(cls, xyz_path: str | Path) -> LiPSStructure:
        if isinstance(xyz_path, str):
            xyz_path = Path(xyz_path)
        assert xyz_path.exists(), f"{xyz_path} not found."
        atoms = read(str(xyz_path), index=":", format="extxyz")
        return cls(*atoms)

    def to_lmdb(self, lmdb_path: str | Path) -> None:
        if isinstance(lmdb_path, str):
            lmdb_path = Path(lmdb_path)
        os.makedirs(lmdb_path, exist_ok=True)
        target_env = lmdb.open(
            str(lmdb_path.joinpath("data.lmdb")),
            subdir=False,
            map_size=1099511627776 * 2,
            meminit=False,
            map_async=True,
        )
        for index, atom in enumerate(
            tqdm(self.atoms, desc="Entries processed", total=len(self)),
        ):
            struct = self.entry_to_dict(atom)
            write_lmdb_data(index, struct, target_env)

    def make_devset(self, random_seed: int = 2150626, num_samples: int = 200) -> None:
        """
        Serialize a devset.

        This is not intended for regular use, and implemented mainly for
        reproducibility.
        The devset will be dumped in this directory, which should then be readily
        accessible through `matsciml.datasets.lips.lips_devset` as a path.

        Parameters
        ----------
        random_seed : int
            Random seed set for shuffling indices
        num_samples : int
            Number of samples to include in the devset, should be fairly small
            (on order of a few hundred)
        """
        from random import seed, shuffle

        # prepare to dump devset
        root = Path(__file__).parent
        # set random seed for shuffling
        seed(random_seed)
        indices = list(range(len(self)))
        shuffle(indices)
        chosen = indices[:num_samples]
        # retrieve the samples, then overwrite the atoms
        atoms = [self.atoms[index] for index in chosen]
        self.atoms = atoms
        self.to_lmdb(root.joinpath("devset"))
