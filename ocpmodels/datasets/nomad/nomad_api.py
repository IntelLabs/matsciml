import multiprocessing
import os
import re
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import suppress
from functools import cached_property
from typing import Dict, List, Optional, Tuple, Union

import requests
import yaml


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
        # elif material_ids is None and split_files is None and split_dir is:
        #     raise Exception(f"Either material_ids or split_files must be specified")

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


if __name__ == "__main__":
    nomad = NomadRequest()
    nomad.fetch_ids()
