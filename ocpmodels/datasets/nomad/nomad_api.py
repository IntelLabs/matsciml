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
import requests
import yaml
from ocpmodels.datasets.utils import write_lmdb_data
from tqdm import tqdm


class NomadRequest:
    base_url = "http://nomad-lab.eu/prod/v1/api/v1"

    id_query = {
        "query": {
            "results.material.structural_type:any": [
                "bulk",
                "molecule / cluster",
                "atom",
            ],
        },
        "pagination": {
            "page_size": 10000,
            "order_by": "upload_create_time",
            "order": "desc",
            "page_after_value": "1689167405976:vwonU6jzh1uruW0S9r9Q_JKz1-O0",
            "next_page_after_value": "1689167405976:vwonU6jzh1uruW0S9r9Q_JKz1-O0",
        },
        "required": {"include": ["entry_id"]},
    }

    results_query = {
        "results.material.structural_type:any": ["bulk", "molecule / cluster"],
        "quantities:all": ["results"],
    }

    def __init__(
        self,
        base_data_dir: str = "./",
        split_files: Optional[List[str]] = None,
        material_ids: Optional[List[int]] = None,
        split_dir: Optional[str] = None,
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
            try:
                self.split_file = os.path.join(self.data_dir, "all.yml")
                ids["all"] = yaml.safe_load(open(self.split_file, "r"))
            except FileNotFoundError:
                raise "Found no split files!"
        return ids

    def fetch_ids(self) -> Dict[int, str]:
        entry_ids = set()
        query_error = False
        from time import time

        p_time = []
        start_time = time()
        while not query_error:
            processing_time = time() - start_time
            p_time.append(processing_time)
            start_time = time()
            response = requests.post(
                f"{NomadRequest.base_url}/entries/query", json=NomadRequest.id_query
            )
            response_json = response.json()
            if response.status_code != 200:
                query_error = True
                print(response.status_code)
                print(response.text)

            ids = [_["entry_id"] for _ in response_json["data"]]
            entry_ids.update(ids)
            NomadRequest.id_query["pagination"]["page_after_value"] = response_json[
                "pagination"
            ]["next_page_after_value"]
            num_entries = len(entry_ids)
            print(
                f"Total IDs: {num_entries}\tThroughput: {round(sum(p_time)/len(p_time) , 3)}",
                end="\r",
            )

        with open(f"{self.split_file}.yml", "w") as f:
            entry_id_dict = dict(zip(range(num_entries), entry_ids))
            yaml.safe_dump(entry_id_dict, f, sort_keys=False)

    def download_data(self):
        """Facilitates downloading data from different splits. Makes a folder
        specifically for the raw .cif files.
        """
        id_dict = self.process_ids()
        for split, ids in id_dict.items():
            self.material_ids = ids
            self.data_dir = os.path.splitext(split)[0]
            print(f"Downloading data to : {self.data_dir}")
            self.nomad_request()

    def fetch_data(self, idx):
        id = self.material_ids[idx]
        response = requests.post(
            f"{NomadRequest.base_url}/entries/{id}/archive/query",
            json=NomadRequest.results_query,
        )
        if response.status_code == 200:
            data = response.json()
            results = data["data"]["archive"]["results"]
            self.data[idx] = results
        else:
            self.request_warning(response, idx)

    def request_warning(self, requested_data, id_idx):
        warning_message = f"Sample {id_idx} from {self.data_dir} failed to download with: {requested_data.status_code}\n"
        warnings.warn(warning_message)
        with open(
            os.path.join(os.path.dirname(self.data_dir), f"failed.txt"), "a"
        ) as f:
            f.write(warning_message)
        return False

    def nomad_request(self):

        self.data = [None] * len(self.material_ids)
        
        with suppress(OSError):
            os.remove(os.path.join(self.data_dir, f"failed.txt"))

        with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
            # List to store the future objects
            futures = [executor.submit(self.fetch_data, n) for n in self.material_ids]

            # Iterate over completed futures to access the responses
            for future in tqdm(
                as_completed(futures), total=len(self.material_ids), desc="Downloading"
            ):
                try:
                    future.result()
                except Exception as e:
                    print(f"Error occurred: {e}")

        self.to_lmdb(self.data_dir)
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


if __name__ == "__main__":
    nomad = NomadRequest()
    nomad.fetch_ids()
    # nomad.data_dir = ""
    # nomad.download_data()
