from __future__ import annotations

import multiprocessing
import os
import re
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import suppress
from functools import cached_property
from typing import Dict, List, Optional, Tuple, Union

import lmdb
import pandas as pd
import requests
import yaml
from tqdm import tqdm

from matsciml.datasets.utils import write_lmdb_data


class CMDRequest:
    def __init__(
        self,
        base_data_dir: str = "./",
        split_files: list[str] | None = None,
        material_ids: list[int] | None = None,
        split_dir: str | None = None,
    ):
        self.split_dir = split_dir
        self.base_data_dir = base_data_dir
        os.makedirs(base_data_dir, exist_ok=True)
        self.material_ids = material_ids
        self.split_files = split_files
        # Can specify material ids' or split_files, but not both. If neither are
        # supplied, the full dataset is downloaded.
        if material_ids and split_files:
            raise Exception(f"Only one of material_ids and split_files may be supplied")
        if split_dir and split_files:
            raise Exception(f"Only one of split_dir and split_files may be supplied")
        elif material_ids is None and split_files is None:
            self.material_ids = list(range(0, 214435))

    @property
    def material_ids(self) -> list[str] | None:
        return self._material_ids

    @material_ids.setter
    def material_ids(self, values: list[int] | None) -> None:
        self._material_ids = values

    @property
    def data_dir(self) -> list[str] | None:
        return self._data_dir

    @data_dir.setter
    def data_dir(self, dst_folder: str | None) -> None:
        """Use the `base_data_dir` plus the destination folder to create `data_dir`.
        The `dst_folder` is determined by which split we are processing specified by
        `split_files`, otherwise will default to `all`.

        Args:
            dst_folder (Union[str, None]): Destination folder for data (raw, and lmdb)
        """
        self._data_dir = os.path.join(self.base_data_dir, dst_folder)
        os.makedirs(self._data_dir, exist_ok=True)

    def process_ids(self) -> dict[str, int]:
        """Builds a dictionary of split names and the id's associated with them.
        If not split files are specified then whatever material id's are present are
        used to create the 'all' split.

        Returns:
            Dict[str, int]: _description_
        """
        ids = {}
        if self.split_files is not None:
            for split_file in self.split_files:
                ids[split_file] = yaml.safe_load(open(split_file))
        if self.split_dir is not None:
            ids[self.split_dir] = self.material_ids
        else:
            ids["all"] = self.material_ids
        return ids

    def download_data(self):
        """Facilitates downloading data from different splits. Makes a folder
        specifically for the raw .cif files.
        """
        id_dict = self.process_ids()
        for split, ids in id_dict.items():
            self.material_ids = ids
            self.data_dir = os.path.join(os.path.splitext(split)[0], "raw_data")
            print(f"Downloading data to : {self.data_dir}")
            self.cmd_request()

    def fetch_data(self, n) -> tuple[int, bool]:
        """Downloads one sample from the CMD database and saves it.

        Args:
            n (_type_): Index of sample to download

        Returns:
            Tuple[int, Dict]: index downloaded, status boolean
        """

        def request_warning(requested_data):
            warning_message = f"Sample {n} from {self.data_dir} failed to download with: {requested_data.status_code}\n"
            warnings.warn(warning_message)
            with open(
                os.path.join(os.path.dirname(self.data_dir), f"failed.txt"),
                "a",
            ) as f:
                f.write(warning_message)
            return False

        cif_url = "http://www.carolinamatdb.org/static/export/cifs/{}.cif"
        energy_url = "http://www.carolinamatdb.org/entry/{}/"
        energy_pattern = (
            r'<h5>Formation Energy / Atom<\/h5>\s*<span class="value">(.*?)<\/span>'
        )
        request_status = []
        time.sleep(0.0001)
        data = requests.get(url=cif_url.format(n))
        energy_data = requests.get(url=energy_url.format(n + 1))

        if data.status_code != 200:
            request_status.append(request_warning(data))

        retry = 0
        while energy_data.status_code != 200 and retry < 5:
            energy_data = requests.get(url=energy_url.format(n + 1))
            time.sleep(1)
            retry += 1

        if energy_data.status_code != 200:
            request_status.append(request_warning(energy_data))

        if data.status_code == 200 and energy_data.status_code == 200:
            data = data.text
            match = re.search(energy_pattern, energy_data.text)
            if match:
                energy = match.group(1).strip()
            else:
                energy = None
            with open(os.path.join(self.data_dir, f"{n}.cif"), "w") as f:
                f.write(data)
                f.write(f"energy {energy}\n")
                f.write(f"origin_file {n}.cif")
        return n, all(request_status)

    def cmd_request(self) -> None:
        """ "Queries the Carolina Materials Database through a specific endpoint.
        Because the database is hosted on a machine with 1 CPU and limited memory, some
        queries may fail. A pause between requests was suggested by the CMD creator.
        If queries fail, a text file with samples that failed will be saved so they can
        be requested again if desired. Uses multiprocessing to speed up downloads.
        """

        request_status = {}

        with suppress(OSError):
            os.remove(os.path.join(self.data_dir, f"failed.txt"))

        with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
            # List to store the future objects
            futures = [executor.submit(self.fetch_data, n) for n in self.material_ids]

            # Iterate over completed futures to access the responses
            for future in tqdm(
                as_completed(futures),
                total=len(self.material_ids),
                desc="Downloading",
            ):
                try:
                    n, status = future.result()
                    request_status[n] = status
                except Exception as e:
                    print(f"Error occurred: {e}")
        return request_status

    @cached_property
    def atomic_number_map(self) -> dict[str, int]:
        """List of element symbols and their atomic numbers.

        Returns:
            Dict[str, int]: _description_
        """
        # fmt: off
        an_map = {
            'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8,
            'F': 9, 'Ne': 10, 'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15,
            'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22,
            'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29,
            'Zn': 30, 'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36,
            'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40, 'Nb': 41, 'Mo': 42, 'Tc': 43,
            'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50,
            'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55, 'Ba': 56, 'La': 57,
            'Ce': 58, 'Pr': 59, 'Nd': 60, 'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64,
            'Tb': 65, 'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70, 'Lu': 71,
            'Hf': 72, 'Ta': 73, 'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78,
            'Au': 79, 'Hg': 80, 'Tl': 81, 'Pb': 82, 'Bi': 83, 'Po': 84, 'At': 85,
            'Rn': 86, 'Fr': 87, 'Ra': 88, 'Ac': 89, 'Th': 90, 'Pa': 91, 'U': 92,
            'Np': 93, 'Pu': 94, 'Am': 95, 'Cm': 96, 'Bk': 97, 'Cf': 98, 'Es': 99,
            'Fm': 100, 'Md': 101, 'No': 102, 'Lr': 103, 'Rf': 104, 'Db': 105,
            'Sg': 106, 'Bh': 107, 'Hs': 108, 'Mt': 109, 'Ds': 110, 'Rg': 111,
            'Cn': 112, 'Nh': 113, 'Fl': 114, 'Mc': 115, 'Lv': 116, 'Ts': 117,
            'Og': 118,
        }
        # fmt: on
        return an_map

    @cached_property
    def files_available(self) -> list[str]:
        files = os.listdir(self.data_dir)
        return files

    def parse_data(self, idx: int) -> tuple[int, dict]:
        """Parse data from .cif file.

        Args:
            idx (int): Index in file list to process

        Returns:
            Tuple[int, Dict]: Index processes, data dictionary
        """
        data = open(os.path.join(self.data_dir, self.files_available[idx])).read()
        data_dict = {}
        # files sometimes come with training blank line
        lines = data.split("\n")
        if lines[-1] == "":
            lines = lines[:-1]
        # remove the header lines:
        lines = lines[2:]
        lines = [line.strip() for line in lines]
        # split lines into property lines, and remainder. note that properties are
        # a mix of string, float and int values, but will all be stored as strings.
        # converting to proper type is done in dataset (not sure if this is best)
        property_lines, lines = (
            lines[: lines.index("loop_")],
            lines[lines.index("loop_") + 1 :],
        )
        props = [property.split(maxsplit=1) for property in property_lines]
        # make a property dictionary and update data_dict
        prop_dict = dict(pd.DataFrame(props).values)
        data_dict.update(prop_dict)
        # split next set of lines to get the symmetry info
        symmetry_lines, lines = (
            lines[lines.index("_symmetry_equiv_pos_as_xyz") + 1 : lines.index("loop_")],
            lines[lines.index("loop_") + 1 :],
        )
        symmetries = pd.DataFrame(
            [symmetry.split(maxsplit=1) for symmetry in symmetry_lines],
        )
        symmetries[0] = pd.to_numeric(symmetries[0])
        symmetry_dict = dict(symmetries.values)
        data_dict["symmetry_dict"] = symmetry_dict
        # split lines to get the cartesian coordinates and atomic numbers
        cart_coords_lines = lines[lines.index("_atom_site_occupancy") + 1 : -2]
        cart_coords_df = pd.DataFrame([p.split() for p in cart_coords_lines])
        cart_coords = cart_coords_df[[3, 4, 5]].to_numpy(dtype=float)
        atomic_numbers = [
            self.atomic_number_map[symbol] for symbol in cart_coords_df[0].values
        ]
        data_dict["atomic_numbers"] = atomic_numbers
        data_dict["cart_coords"] = cart_coords
        data_dict["energy"] = float(lines[-2].split(maxsplit=1)[-1])
        data_dict["formula_pretty"] = data_dict["_chemical_formula_sum"].replace(
            " ",
            "",
        )
        data_dict["origin_file"] = lines[-1].split(maxsplit=1)[-1]
        return idx, data_dict

    def process_data(self) -> dict:
        """Processes the raw .cif data. Grabbing any properties or attributes that are
        provided, as well as the position data. Gathers everything into a dictionary,
        and then saves to LMDB at the end. Uses multiprocessing to speed up processing.

        Returns:
            Dict: the processed data
        """
        self.data = [None] * len(self.files_available)

        with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
            # List to store the future objects
            futures = [
                executor.submit(self.parse_data, n) for n in range(len(self.data))
            ]

            # Iterate over completed futures to access the responses
            for future in tqdm(
                as_completed(futures),
                total=len(self.data),
                desc="Parsing Raw Data",
            ):
                try:
                    idx, parsed_data = future.result()
                    self.data[idx] = parsed_data
                except Exception as e:
                    print(f"Error occurred: {e}")

        self.to_lmdb(os.path.dirname(self.data_dir))
        return self.data

    def to_lmdb(self, lmdb_path: str) -> None:
        """
        Save the retrieved documents to an LMDB file.

        Requires specifying a folder to save to, in which a "data.lmdb" file
        and associated lockfile will be created. Each entry is saved as key/
        value pairs, with keys simply being the index, and the value being
        a pickled dictionary representation of the retrieved `SummaryDoc`.

        Parameters
        ----------
        lmdb_path : str
            Directory to save the LMDB data to.

        Raises
        ------
        ValueError:
            [TODO:description]
        """
        os.makedirs(lmdb_path, exist_ok=True)
        target_env = lmdb.open(
            os.path.join(lmdb_path, "data.lmdb"),
            subdir=False,
            map_size=1099511627776 * 2,
            meminit=False,
            map_async=True,
        )
        if self.data is not None:
            for index, entry in tqdm(
                enumerate(self.data),
                desc="Entries processed",
                total=len(self.data),
            ):
                write_lmdb_data(index, entry, target_env)
        else:
            raise ValueError(
                f"No data was available for serializing - did you run `retrieve_data`?",
            )

    @classmethod
    def make_devset(cls):
        kwargs = {
            "base_data_dir": "./matsciml/datasets/carolina_db/",
            "material_ids": list(range(0, 100)),
        }
        cmd = cls(**kwargs)
        cmd.data_dir = "devset"
        cmd.cmd_request()
        cmd.process_data()
