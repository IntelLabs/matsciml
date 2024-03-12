from __future__ import annotations

import bz2
import json
import os
from typing import Any, Optional

import numpy as np
import requests
from joblib import Parallel, delayed
from tqdm import tqdm

from matsciml.datasets import utils


class AlexandriaRequest:
    def __init__(
        self,
        indices: list[int],
        lmdb_target_dir: str,
        dataset: str = "3D",
        target_keys: list[str] = [
            "energy_total",
            "total_mag",
            "dos_ef",
            "band_gap_ind",
            "e_form",
            "e_above_hull",
        ],
        devset: bool = False,
        max_atoms: Optional[int] = 10000,
    ):
        """
        Download a set of files from a dataset of the Alexandria database and write it to a set of LMDB file.
        All single atom structures are removed to avoid errors in dgl.

        Parameters
        ----------
        indices: int
            Indices of the file to download. 3D 0-44, 2D 0-1, 1d 0, scan/pbesol 0-4
        lmdb_target_dir : str
            Path to the directory where the LMDB file will be written.
        dataset : str
            Name of the dataset to download from. Must be one of '1D', '2D', '3D', 'scan' or 'pbesol'.
            The 2D and 1D structures are peridioc in the "non-periodic" directions with a vacuum of 15 Å.
            Cutoff distances during graph construction larger than this vacuum will  produce wrong neighborlists.
        target_keys : list[str]
            List of target keys to extract from the downloaded file. Atomic magnetic moments and
            forces will be added separately as they are per atom properties.
        devset : bool
            boolean to indicate if the devset should be downloaded. In this case only the first 100 entries
            will be written to an LMDB file.
        max_atoms : int
            Maximum number of atoms in the structure. Structures with more atoms are removed during the download.
            This can be used to avoid unbalanced batchsizes during training.
        Raises
        ------
        ValueError
            If the dataset name is not one of '1D', '2D', '3D', 'scan' or 'pbesol'.
        """
        self.target_keys = target_keys
        self.dataset = dataset
        self.target_dir = lmdb_target_dir
        self.devset = devset
        self.max_atoms = max_atoms
        if not os.path.isdir(self.target_dir):
            os.mkdir(self.target_dir)

        base_URL = "https://alexandria.icams.rub.de/data"
        urls = []
        for index in indices:
            index = str(index)
            if len(index) == 1:
                index = "0" + index
            if dataset == "3D":
                URL = base_URL + "/pbe/alexandria_0" + index + ".json.bz2"
            elif dataset == "2D":
                URL = base_URL + "/pbe_2d/alexandria_2d_0" + index + ".json.bz2"
            elif dataset == "1D":
                URL = base_URL + "/pbe_1d/alexandria_1d_0" + index + ".json.bz2"
            elif dataset == "scan":
                URL = base_URL + "/scan/alexandria_scan_0" + index + ".json.bz2"
            elif dataset == "pbesol":
                URL = base_URL + "/pbesol/alexandria_ps_0" + index + ".json.bz2"
            else:
                raise ValueError("Dataset must be 1D, 2D, 3D, scan or pbesol")

            urls.append(URL)
        self.urls = urls

    def get_data_dict(self, computed_entry_dict: dict[str, Any]) -> dict[str, Any]:
        """
        Processes the ComputedStructureEntry dictionary to extract the structure, forces and magnetic moments
        and other target properties.
        """

        def get_forces_array_from_structure(structure):
            forces = [site["properties"]["forces"] for site in structure["sites"]]
            return np.array(forces)

        def get_magmoms_array_from_structure(structure):
            magmoms = [site["properties"]["magmom"] for site in structure["sites"]]
            return np.array(magmoms)

        targets = {
            "regression": {
                target: computed_entry_dict["data"][target]
                for target in self.target_keys
            },
            "classification": {},
        }
        structure = computed_entry_dict["structure"]
        data = dict(
            structure=structure,
            force=get_forces_array_from_structure(structure),
            entry_id=computed_entry_dict["data"]["mat_id"],
            natoms=len(structure["sites"]),
            magmoms=get_magmoms_array_from_structure(structure),
            targets=targets,
        )
        return data

    def process_index(self, index: int) -> None:
        """
        Download a file from a dataset of the Alexandria database with the respective index
        and write it to the LMDB file with the respective index.

        Parameters
        ----------
        index : int
            Index of the file to download. 3D 0-44, 2D 0-1, 1d 0, scan/pbesol 0-4
        """

        def download(url):
            response = requests.get(url)
            # Check if the request was successful
            if response.status_code == 200:
                decompressed_data = bz2.decompress(response.content)
                json_str = decompressed_data.decode("utf-8")
                data = json.loads(json_str)
                return data
            else:
                raise ValueError(
                    f"Request failed with status code {response.status_code}",
                )

        data = download(self.urls[index])
        if self.devset:
            data["entries"] = data["entries"][0:100]

        computed_entry_dict = [
            self.get_data_dict(entry)
            for entry in tqdm(data["entries"], desc=f"Processing file {index}")
        ]
        lmdb_env = utils.connect_lmdb_write(
            os.path.join(self.target_dir, "data_00" + str(index)),
        )
        for subindex, _data in enumerate(
            tqdm(
                computed_entry_dict,
                desc=f"Writing LMDB data to file {lmdb_env.path()}",
            ),
        ):
            # remove single atom structures to avoid errors in dgl
            if _data["natoms"] > 1 and _data["natoms"] < self.max_atoms:
                utils.write_lmdb_data(subindex, _data, lmdb_env)

    def download_and_write(self, n_jobs: int = 1) -> None:
        """
        Download a set of files from a dataset of the Alexandria database and write it to a set of LMDB file.

        Parameters
        ----------
        n_jobs : int
            Number of parallel jobs to run. Default is 1.
        """
        _ = Parallel(n_jobs=n_jobs, backend="multiprocessing", verbose=10)(
            delayed(self.process_index)(index) for index in range(len(self.urls))
        )

    @classmethod
    def devset(cls, lmdb_target_dir: str) -> AlexandriaRequest:
        return cls([0], lmdb_target_dir, dataset="3D", devset=True)
