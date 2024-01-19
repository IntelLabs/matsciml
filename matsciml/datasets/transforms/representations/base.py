from __future__ import annotations

from abc import abstractmethod
from typing import List

from matsciml.common import DataDict
from matsciml.common.types import DataDict
from matsciml.datasets.transforms.base import AbstractDataTransform


class RepresentationTransform(AbstractDataTransform):
    """
    Transforms under this class are designed to convert one data structure
    to another; for example, a point cloud to a graph and vice versa.

    To subclass, we have prologue, convert, and epilogue methods to
    override, with convert being the only method that _needs_ to be
    changed. The prologue is only run once at the beginning, mainly
    to make sure that users don't apply the wrong transform.
    """

    def __init__(self, backend: str) -> None:
        super().__init__()
        # this flag can be used to determine if prologue is run
        self._has_started = False
        self.backend = backend

    @property
    def backend(self) -> str:
        return self._backend

    @backend.setter
    def backend(self, value: str) -> None:
        assert value in [
            "dgl",
            "pyg",
        ], f"Backend {value} not supported. Must be 'dgl' or 'pyg'."
        self._backend = value

    def prologue(self, data: DataDict) -> None:
        self._has_started = True

    @abstractmethod
    def convert(self, data: DataDict) -> None:
        ...

    def epilogue(self, data: DataDict) -> None:
        # used to clean up the DataDict before returning it
        # e.g. to remove graph keys
        pass

    @staticmethod
    def _check_for_type(data: DataDict, _types: list[type]) -> bool:
        """
        Checks whether there are entries within a DataDict structure that
        contain any of the types provided.

        Parameters
        ----------
        data : DataDict
            Dict containing a single data sample
        _types : List[type]
            List of types to check against

        Returns
        -------
        bool
            True if any of the entries are an instance of any of the
            types provided
        """
        return any([isinstance(entry, _types) for entry in data.values()])

    def __call__(self, data: DataDict) -> DataDict:
        """
        Perform an in-place transformation of one data representation
        to another.

        Parameters
        ----------
        data : DataDict
            Dict cotaining a single data sample

        Returns
        -------
        DataDict
            Modified data sample with the new representation
        """
        # do things like type checking
        if not self._has_started:
            self.prologue(data)
        self.convert(data)
        self.epilogue(data)
        return data
