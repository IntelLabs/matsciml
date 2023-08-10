import multiprocessing
import os
import re
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import suppress
from functools import cached_property
from typing import Dict, List, Optional, Union, Tuple

import lmdb
import pandas as pd
import requests
import yaml
from tqdm import tqdm

from ocpmodels.datasets.utils import write_lmdb_data


class OQMDRequest:
    def __init__(
        self,
        base_data_dir: str = "./",
        split_files: Optional[List[str]] = None,
        material_ids: Optional[List[int]] = None,
        split_dir: Optional[str] = None,
        limit: int = 1000,
    ):
        self.split_dir = split_dir
        self.base_data_dir = base_data_dir
        os.makedirs(base_data_dir, exist_ok=True)
        self.material_ids = material_ids
        self.split_files = split_files
        self.limit = limit
        # Can specify material ids' or split_files, but not both. If neither are
        # supplied, the full dataset is downloaded.
        if material_ids and split_files:
            raise Exception(f"Only one of material_ids and split_files may be supplied")
        if split_dir and split_files:
            raise Exception(f"Only one of split_dir and split_files may be supplied")
        elif material_ids is None and split_files is None:
            self.material_ids = list(range(0, 214435))

    @property
    def material_ids(self) -> Union[List[str], None]:
        return self._material_ids

    @material_ids.setter
    def material_ids(self, values: Union[List[int], None]) -> None:
        self._material_ids = values

    @property
    def data_dir(self) -> Union[List[str], None]:
        return self._data_dir

    @data_dir.setter
    def data_dir(self, dst_folder: Union[str, None]) -> None:
        """Use the `base_data_dir` plus the destination folder to create `data_dir`.
        The `dst_folder` is determined by which split we are processing specified by
        `split_files`, otherwise will default to `all`.

        Args:
            dst_folder (Union[str, None]): Destination folder for data (raw, and lmdb)
        """
        self._data_dir = os.path.join(self.base_data_dir, dst_folder)
        os.makedirs(self._data_dir, exist_ok=True)

    def process_ids(self) -> Dict[str, int]:
        """Builds a dictionary of split names and the id's associated with them.
        If not split files are specified then whatever material id's are present are
        used to create the 'all' split.

        Returns:
            Dict[str, int]: _description_
        """
        ids = {}
        if self.split_files is not None:
            for split_file in self.split_files:
                ids[split_file] = yaml.safe_load(open(split_file, "r"))
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
            self.data_dir = os.path.splitext(split)[0]
            print(f"Downloading data to : {self.data_dir}")
            self.oqmd_request()

    def oqmd_request(self) -> None:
        def parse_sites(sites):
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

        def request_warning(requested_data):
            warning_message = f"Offset {idx*self.limit} failed to download with: {requested_data.status_code}\n"
            warnings.warn(warning_message)
            with open(
                os.path.join(os.path.dirname(self.data_dir), f"failed.txt"), "a"
            ) as f:
                f.write(warning_message)
            return False

        request_status = {}
        oqmd_url = "http://oqmd.org/oqmdapi/formationenergy?&limit={}&offset={}"
        oqmd_data = []

        idx = 0
        data = [None] * self.limit
        while len(data) == self.limit:
            time.sleep(0.0001)
            data = requests.get(url=oqmd_url.format(self.limit, idx * self.limit))
            retry = 0
            while data.status_code != 200 and retry < 5:
                data = requests.get(url=oqmd_url.format(self.limit, idx * self.limit))
                time.sleep(1)
                retry += 1

            if data.status_code == 200:
                data = data.json()
                data = data["data"]
                for n in range(len(data)):
                    data[n]["atomic_numbers"], data[n]["cart_coords"] = parse_sites(
                        data[n]["sites"]
                    )
                    data[n].pop("sites")
                oqmd_data.extend(data)
            else:
                request_status = request_warning(data)
            idx += 1
            print(len(oqmd_data), end='\r')

        self.to_lmdb(self.data_dir)
        return request_status

    @cached_property
    def atomic_number_map(self) -> Dict[str, int]:
        """List of element symbols and their atomic numbers.

        Returns:
            Dict[str, int]: _description_
        """
        # fmt: off
        an_map = {'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 
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
              'Og': 118}
        # fmt: on
        return an_map

    def process_data(self) -> Dict:
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
                as_completed(futures), total=len(self.data), desc="Parsing Raw Data"
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
                enumerate(self.data), desc="Entries processed", total=len(self.data)
            ):
                write_lmdb_data(index, entry, target_env)
        else:
            raise ValueError(
                f"No data was available for serializing - did you run `retrieve_data`?"
            )

    @classmethod
    def make_devset(cls):
        kwargs = {
            "base_data_dir": "./ocpmodels/datasets/oqmd/",
            "material_ids": list(range(0, 100)),
        }
        oqmd = cls(**kwargs)
        oqmd.data_dir = "devset"
        oqmd.oqmd_request()
        oqmd.process_data()


if __name__ == "__main__":
    oqmd = OQMDRequest(base_data_dir="./ocpmodels/datasets/oqmd", limit=100)
    oqmd.download_data()

    # import requests
    # limit = 1000
    # offset = 1000
    # data = requests.get(
    #     url=f"http://oqmd.org/oqmdapi/formationenergy?&limit={limit}&offset={offset}"
    # )
    # # data = data.json()
    # data["data"][0]["entry_id"]
    # data["data"][1]["entry_id"]
