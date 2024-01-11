from __future__ import annotations

import os
from datetime import datetime
from functools import cached_property
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import lmdb
import numpy as np
import yaml
from emmet.core.summary import SummaryDoc
from mp_api.client import MPRester
from tqdm import tqdm

from matsciml.datasets.materials_project.utils import get_split_map
from matsciml.datasets.utils import write_lmdb_data


class MaterialsProjectRequest:
    def __init__(
        self,
        fields: list[str],
        api_key: str | None = None,
        material_ids: list[str] | None = None,
        **api_kwargs,
    ):
        api_kwargs.setdefault("chunk_size", 1000)
        api_kwargs.setdefault("num_sites", (2, 1000))
        # set the API key, which will make sure we one is provided
        # before we check the fields
        self.api_key = api_key
        self.fields = fields
        self.material_ids = material_ids
        self.api_kwargs = api_kwargs

    @property
    def material_ids(self) -> list[str] | None:
        return self._material_ids

    @material_ids.setter
    def material_ids(self, values: list[str] | None) -> None:
        self._material_ids = values

    @property
    def api_key(self) -> str:
        return self._api_key

    @api_key.setter
    def api_key(self, value: str | None = None) -> None:
        """
        Sets the Materials Project API key.

        This method will either use the provided value, or read from the environment
        variable "MP_API_KEY"; the latter is the preferred/recommended method for
        specifying the key.

        Parameters
        ----------
        value : str
            API key value for Materials Project

        Raises
        ------
        ValueError:
            If no API key is provided or found in the environment variable,
            throw a `ValueError`.
        """
        if not value:
            value = os.getenv("MP_API_KEY", None)
        if not value:
            raise ValueError(
                f"No Materials Project API key provided or found in environment variable MP_API_KEY.",
            )
        self._api_key = value

    @property
    def fields(self) -> list[str]:
        return list(self._fields)

    @fields.setter
    def fields(self, values: list[str] | None = None) -> None:
        """
        Method for setting which fields to query.

        In this method, the "structure" field is always requested to ensure that
        we always have something to work with.

        Parameters
        ----------
        values : Union[List[str], None], by default None
            List of field names to query the Materials Project API.
        """
        # first deal with the case with no fields; we need structure at the very least
        if values is None:
            values = [
                "structure",
            ]
        else:
            # remove duplicates
            values = set(values)
            # need to make sure we always have structure as a request
            values.add("structure")
            # check to make sure all of the requested keys exist
            for key in values:
                assert (
                    key in self.available_fields
                ), f"{key} is not a valid field in Materials Project: {self.available_fields}"
        self._fields = values

    def _api_context(self) -> MPRester:
        """
        Simple reusable wrapper for the `MPRester` for API calls.

        Returns
        -------
        MPRester
            Instance of `MPRester` with API key used
        """
        return MPRester(self.api_key)

    @cached_property
    def available_fields(self) -> list[str]:
        """
        Queries and caches a list of available fields from Materials project.

        This is used to validate the requested fields before actually
        starting to download data.

        Returns
        -------
        List[str]
            List of field names available via the API.
        """
        with self._api_context() as mpr:
            return mpr.summary.available_fields

    def retrieve_data(self) -> list[SummaryDoc] | list[dict[Any, Any]]:
        """
        Execute the API requests.

        This will use the specified fields and material IDs to query the
        Materials project API, and return data entries.

        Returns
        -------
        Union[List[SummaryDoc], List[Dict[Any, Any]]]
            Returned entries for the query.
        """
        with self._api_context() as mpr:
            # todo allow material ids to be specified
            docs = mpr.summary.search(
                fields=self.fields,
                material_ids=self.material_ids,
                **self.api_kwargs,
            )
        self.data = docs
        self.retrieved = str(datetime.now())
        return docs

    @property
    def data(self) -> list[SummaryDoc] | list[dict[Any, Any]] | None:
        return getattr(self, "_data")

    @data.setter
    def data(self, values: list[SummaryDoc] | list[dict[Any, Any]]) -> None:
        self._data = values

    @classmethod
    def devset(cls, api_key: str | None = None) -> MaterialsProjectRequest:
        kwargs = {
            "num_elements": (1, 2),
            "num_chunks": 2,
            "chunk_size": 100,
            "num_sites": (2, 100),
        }
        return cls(["band_gap", "structure"], api_key, **kwargs)

    def make_splits(
        self,
        data_dir: str,
        split_value: str = "crystal_class",
        split_percents: dict[str, float] = {"train": 0.7, "val": 0.2, "test": 0.1},
    ):
        """Make data splits for train, test and validation

        Args:
            data_dir (str): Directory to save the output lmdb's.
            split_value (str, optional): Property used to make the splits. Defaults to "crystal_class".
            split_percents (_type_, optional): Distribution to use for train, test and val splits. Defaults to {"train": 0.7, "val": 0.2, "test": 0.1}.
        """
        split_percent_values = list(split_percents.values())
        indices = {k: [] for k in split_percents.keys()}

        split_map = get_split_map(self.data, split_value)

        for prop_key, prop_idx in split_map.items():
            prop_idx = np.random.permutation(prop_idx)
            split_idx = [
                int(sum(split_percent_values[:idx]) * len(prop_idx))
                for idx in range(1, len(split_percent_values))
            ]
            data_splits = np.split(prop_idx, split_idx)
            sub_dict = dict(zip(indices.keys(), data_splits))
            for sub_k, sub_v in sub_dict.items():
                indices[sub_k].extend(list(sub_v))

        og_data = self.data.copy()
        for split_name, index_list in indices.items():
            id_list = []
            self.data = []
            for idx in index_list:
                id_list.append(og_data[idx].material_id.string)
                self.data.append(og_data[idx])

            with open(os.path.join(data_dir, split_name + ".yml"), "w") as f:
                yaml.dump({split_name: id_list}, f, sort_keys=False)
            self.to_lmdb(os.path.join(data_dir, split_name))

    def to_lmdb(self, lmdb_path: str | Path) -> None:
        """
        Save the retrieved documents to an LMDB file.

        Requires specifying a folder to save to, in which a "data.lmdb" file
        and associated lockfile will be created. Each entry is saved as key/
        value pairs, with keys simply being the index, and the value being
        a pickled dictionary representation of the retrieved `SummaryDoc`.

        Parameters
        ----------
        lmdb_path : Union[str, Path]
            Directory to save the LMDB data to.

        Raises
        ------
        ValueError:
            [TODO:description]
        """
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
        if self.data is not None:
            for index, entry in tqdm(
                enumerate(self.data),
                desc="Entries processed",
                total=len(self.data),
            ):
                write_lmdb_data(index, entry.__dict__, target_env)
        else:
            raise ValueError(
                f"No data was available for serializing - did you run `retrieve_data`?",
            )

    def to_dict(self) -> dict[str, str | list[str]]:
        """
        Export a summary of the request into a JSON serializable format.

        Returns
        -------
        Dict[str, Union[str, List[str]]]
            Key/value mapping of parameters used and metadata
        """
        data = {}
        data["fields"] = self.fields
        data["material_ids"] = self.material_ids
        data["available_fields"] = self.available_fields
        data["retrieved"] = getattr(self, "retrieved", None)
        data["api_kwargs"] = getattr(self, "api_kwargs", None)
        return data
