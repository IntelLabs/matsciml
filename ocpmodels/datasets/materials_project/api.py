from __future__ import annotations
from typing import List, Optional, Union, Any, Dict
from functools import cached_property
import os

from emmet.core.summary import SummaryDoc
from mp_api.client import MPRester


class MaterialsProjectRequest:
    def __init__(
        self,
        fields: List[str],
        api_key: Optional[str] = None,
        material_ids: Optional[List[str]] = None,
        **api_kwargs,
    ):
        api_kwargs.setdefault("chunk_size", 1000)
        # set the API key, which will make sure we one is provided
        # before we check the fields
        self.api_key = api_key
        self.fields = fields
        self.material_ids = material_ids
        self.api_kwargs = api_kwargs

    @property
    def material_ids(self) -> Union[List[str], None]:
        return self._material_ids

    @material_ids.setter
    def material_ids(self, values: Union[List[str], None]) -> None:
        self._material_ids = values

    @property
    def api_key(self) -> str:
        return self._api_key

    @api_key.setter
    def api_key(self, value: Optional[str] = None) -> None:
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
                f"No Materials Project API key provided or found in environment variable MP_API_KEY."
            )
        self._api_key = value

    @property
    def fields(self) -> Union[List[str], None]:
        return self._fields

    @fields.setter
    def fields(self, values: Union[List[str], None] = None) -> None:
        if values:
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
    def available_fields(self) -> List[str]:
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

    def retrieve_data(self) -> Union[List[SummaryDoc], List[Dict[Any, Any]]]:
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
                fields=self.fields, material_ids=self.material_ids, **self.api_kwargs
            )
        self.data = docs
        return docs

    @property
    def data(self) -> Union[List[SummaryDoc], List[Dict[Any, Any]], None]:
        return getattr(self, "_data")

    @data.setter
    def data(self, values: Union[List[SummaryDoc], List[Dict[Any, Any]]]) -> None:
        self._data = values

    @classmethod
    def devset(cls, api_key: Optional[str] = None) -> MaterialsProjectRequest:
        kwargs = {"num_elements": (1, 2), "num_chunks": 2, "chunk_size": 100}
        return cls(["band_gap", "structure"], api_key, **kwargs)
