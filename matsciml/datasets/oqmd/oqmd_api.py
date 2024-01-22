from __future__ import annotations

import json
import multiprocessing
import os
import random
import warnings
from functools import cached_property
from time import sleep, time
from typing import Dict, List, Optional, Tuple, Union

import lmdb
import requests
import yaml
from tqdm import tqdm

from matsciml.datasets.utils import write_lmdb_data


class QueryIndex:
    """Custom iterator used to facilitate downloading data either by ID or until there
    is no more data to download.
    """

    def __init__(self, ids: list[int] = None):
        if ids is not None:
            self.custom_ids = True
            self.ids = ids + [None]
        else:
            self.custom_ids = False
            self.ids = 0

    def __next__(self):
        try:
            if self.custom_ids:
                return_value = self.ids.pop(0)
            else:
                return_value = self.ids
                self.ids += 1
            return return_value
        except IndexError:
            raise StopIteration

    def __iter__(self):
        return self


class OQMDRequest:
    """Query data from the Open Quantum Materials Database using their REST API.
    To download all properties of all materials, the url http://oqmd.org/oqmdapi/formationenergy
    is used. See https://static.oqmd.org/static/docs/restful.html for more details.
    """

    def __init__(
        self,
        base_data_dir: str = "./",
        split_files: list[str] | None = None,
        material_ids: list[int] | None = None,
        split_dir: str | None = None,
        limit: int = 1000,
        num_workers: int = 1,
    ):
        self.split_dir = split_dir
        self.base_data_dir = base_data_dir
        os.makedirs(base_data_dir, exist_ok=True)
        self.material_ids = material_ids
        self.split_files = split_files
        self.limit = limit
        if num_workers == -1:
            num_workers = multiprocessing.cpu_count()
        self.num_workers = num_workers
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
        specifically for the raw files stored in json format.
        """
        id_dict = self.process_ids()
        for split, ids in id_dict.items():
            self.material_ids = ids
            self.data_dir = os.path.join(os.path.splitext(split)[0], "raw_data")
            print(f"Downloading data to : {self.data_dir}")
            self.oqmd_request()

    def parse_sites(self, sites: list[str]) -> tuple[list, list]:
        """Extract `cart_coords` from `sites`, which are a list of strings with
        element name followed by x, y, z coordinates, e.g. "Na @ 0.1 -0.1 0.3".

        Args:
            sites (List[str]): List of sites

        Returns:
            Tuple[List, List]: List of atomic numbers, and list of cartesian positions.
        """
        symbols = []
        cart_coords = []
        for entry in sites:
            parts = entry.split("@")
            if len(parts) != 2:
                print(f"Invalid entry: {entry}")
                continue

            symbol = parts[0].strip()
            symbols.append(symbol)

            coords_str = parts[1].strip()
            coords = [float(coord) for coord in coords_str.split()]
            if len(coords) != 3:
                print(f"Invalid coordinates: {coords_str}")
                continue

            cart_coords.append(coords)
        atomic_numbers = [self.atomic_number_map[symbol] for symbol in symbols]
        return atomic_numbers, cart_coords

    def oqmd_request(self) -> dict[int, bool]:
        """Query the OQMD API which uses pagination to supply data, which is selected by
        a page limit and offset. When querying the whole database (material_ids=None),
        this will run unitl no more data is returned. When using material_ids, they
        should be specified by their offset, and use a page limit of 1, meaning samples
        will be downloaded individually. Data is dumped into json files that are
        processed further down the line. This makes it easy to pick up on an offset that
        may have failed and broken the pipeline.

        Returns:
            Dict[bool]: request status of each query.
        """

        def request_warning(requested_data):
            warning_message = f"Offset {index*self.limit} failed to download with: {requested_data.status_code}\n"
            warnings.warn(warning_message)
            with open(
                os.path.join(os.path.dirname(self.data_dir), f"failed.txt"),
                "a",
            ) as f:
                f.write(warning_message)
            return False

        request_status = {}
        oqmd_url = "http://oqmd.org/oqmdapi/formationenergy?&limit={}&offset={}"

        query_index = QueryIndex(self.material_ids)
        index = next(query_index)
        has_more_data = True
        retry = 0
        while has_more_data and retry < 10 and index is not None:
            t1 = time()
            data = requests.get(url=oqmd_url.format(self.limit, index * self.limit))
            retry = 0
            while data.status_code != 200 and retry < 10:
                data = requests.get(url=oqmd_url.format(self.limit, index * self.limit))
                sleep(60)
                retry += 1

            if data.status_code == 200:
                data = data.json()
                request_status[index] = True
                for n in range(len(data["data"])):
                    (
                        data["data"][n]["atomic_numbers"],
                        data["data"][n]["cart_coords"],
                    ) = self.parse_sites(data["data"][n]["sites"])
                    data["data"][n].pop("sites")
                if data["meta"]["more_data_available"]:
                    has_more_data = True
                else:
                    has_more_data = False
            else:
                request_status[index] = request_warning(data)

            t2 = time()
            print(f"Loading Batch {index} time {round(t2 - t1, 2)} seconds")
            with open(
                os.path.join(self.data_dir, "query_" + str(index) + ".json"),
                "w",
            ) as json_file:
                json.dump(data["data"], json_file, indent=2)
            index = next(query_index)
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

    def process_json(self):
        """Process json files to be saved into Open MatSciML format (lmdb files). A
        set of required keys are defined to ensure all data have the same contents.
        """
        files = os.listdir(self.data_dir)
        required_keys = {
            "name",
            "entry_id",
            "calculation_id",
            "icsd_id",
            "formationenergy_id",
            "duplicate_entry_id",
            "composition",
            "composition_generic",
            "prototype",
            "spacegroup",
            "volume",
            "ntypes",
            "natoms",
            "unit_cell",
            "band_gap",
            "delta_e",
            "stability",
            "fit",
            "calculation_label",
            "atomic_numbers",
            "cart_coords",
        }
        oqmd_data = []
        for file in tqdm(files, desc="Processing Json Files"):
            with open(os.path.join(self.data_dir, file)) as f:
                try:
                    data = json.load(f)
                except Exception:
                    print(os.path.join(self.data_dir, file))
                    continue

                for n in range(len(data)):
                    if (
                        not data[n].get("cart_coords", False)
                        and data[n].get("sites") is not None
                    ):
                        try:
                            (
                                data[n]["atomic_numbers"],
                                data[n]["cart_coords"],
                            ) = self.parse_sites(data[n]["sites"])
                        except KeyError:
                            print(f"Key error in file {file}")

                    sample_keys = list(data[n].keys())
                    present = [key in sample_keys for key in required_keys]

                    if all(present):
                        oqmd_data.append(data[n])
                    else:
                        pass

        self.data = oqmd_data
        return

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
        random.seed(6)
        kwargs = {
            "base_data_dir": "./matsciml/datasets/oqmd/",
            "material_ids": list(random.sample(range(0, 1000000), 100)),
            "limit": 1,
        }
        oqmd = cls(**kwargs)
        oqmd.data_dir = "devset"
        oqmd.oqmd_request()
        oqmd.process_json()
        oqmd.to_lmdb(oqmd.data_dir)


if __name__ == "__main__":
    OQMDRequest.make_devset()
    # oqmd = OQMDRequest(
    #     base_data_dir="./matsciml/datasets/oqmd", limit=100, num_workers=1
    # )
    # oqmd.download_data()
    # oqmd.base_data_dir = "./"
    # oqmd.data_dir = "query_files"
    # oqmd.process_json()
