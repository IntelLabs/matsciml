# Copyright (C) 2022-2023 Intel Corporation
# SPDX-License-Identifier: MIT License

from typing import Union, List, Any, Tuple, Callable, Optional, Dict
from pathlib import Path
from abc import abstractmethod
from random import sample
import functools
import pickle

import lmdb
import torch
from torch.utils.data import Dataset, DataLoader
import dgl
from dgl.nn.pytorch.factory import KNNGraph

from ocpmodels.common.types import DataDict, BatchDict
from ocpmodels.datasets import utils


# this provides some backwards compatiability to Python ~3.7
if hasattr(functools, "cache"):
    from functools import cache
else:
    from functools import lru_cache

    cache = lru_cache(maxsize=None)


def open_lmdb_file(path: Union[str, Path], **kwargs) -> lmdb.Environment:
    """
    Minimally opinionated way of opening LMDB files; by default will
    just be in readonly to prevent accidental writes, as well as
    assume that the path _contains_ LMDB files, and is not an LMDB
    file itself.

    Parameters
    ----------
    path : Union[str, Path]
        Path to the folder containing LMDB files

    Returns
    -------
    lmdb.Environment
        `Environment` object for accessing the data
    """
    kwargs.setdefault("readonly", True)
    kwargs.setdefault("subdir", False)
    if isinstance(path, Path):
        path = str(path)
    return lmdb.open(path, **kwargs)


def read_lmdb_file(path: Union[str, Path], **kwargs) -> lmdb.Environment:
    """
    Sets up opinionated defaults for _reading_ LMDB files, particularly
    used for the Dataset loading.

    Parameters
    ----------
    path : Union[str, Path]
        Path to a single `.lmdb` file.

    Returns
    -------
    lmdb.Environment
        `Environment` object for accessing the data
    """
    kwargs.setdefault("readonly", True)
    kwargs.setdefault("lock", False)
    kwargs.setdefault("subdir", False)
    kwargs.setdefault("readahead", False)
    kwargs.setdefault("max_readers", 1)
    kwargs.setdefault("meminit", False)
    return open_lmdb_file(path, **kwargs)


class BaseLMDBDataset(Dataset):
    """
    Main purpose of this class is to inherit LMDB file
    reading.
    """

    __devset__ = None

    def __init__(
        self,
        lmdb_root_path: Union[str, Path],
        transforms: Optional[List[Callable]] = None,
    ) -> None:
        super().__init__()
        if isinstance(lmdb_root_path, str):
            lmdb_root_path = Path(lmdb_root_path)
        # check that the root path exists
        assert (
            lmdb_root_path.exists()
        ), f"Root folder for dataset does not exist {lmdb_root_path}"
        # check LMDB files exist within the subdirectory
        db_paths = sorted(lmdb_root_path.glob("*.lmdb"))
        assert len(db_paths) > 0, f"No LMDBs found in '{lmdb_root_path}'"
        self._envs = [read_lmdb_file(path) for path in db_paths]
        self.transforms = transforms

    @property
    def transforms(self) -> List[Callable]:
        return self._transforms

    @transforms.setter
    def transforms(self, values: Union[List[Callable], None]) -> None:
        # if transforms are passed, this gives an opportunity for
        # each transform to modify the state of this dataset
        if values:
            for transform in values:
                if hasattr(transform, "setup_transform"):
                    transform.setup_transform(self)
        self._transforms = values

    @property
    @abstractmethod
    def data_loader(self) -> DataLoader:
        raise NotImplementedError(
            f"No data loader specified for {self.__class__.__name__}."
        )

    @cache
    def _load_keys(self) -> List[Tuple[int, int]]:
        """
        Load in all of the indices from each LMDB file. This creates an
        easy lookup of which data point is mapped to which total dataset
        index, as the former is returned as a simple 2-tuple of lmdb
        file index and the subindex (i.e. the actual data to read in).

        Returns
        -------
        List[Tuple[int, int]]
            2-tuple of LMDB file index and data index within the file
        """
        indices = []
        for lmdb_index, env in enumerate(self._envs):
            with env.begin() as txn:
                # this gets all the keys within the LMDB file, including metadata
                lmdb_keys = [
                    value.decode("utf-8")
                    for value in txn.cursor().iternext(values=False)
                ]
                # filter out non-numeric keys
                subindices = filter(lambda x: x.isnumeric(), lmdb_keys)
                indices.extend([(lmdb_index, int(subindex)) for subindex in subindices])
        return indices

    def index_to_key(self, index: int) -> Tuple[int]:
        """For trajectory dataset, just grab the 2-tuple of LMDB index and subindex"""
        return self.keys[index]

    def data_from_key(self, lmdb_index: int, subindex: int) -> Any:
        """
        Retrieve a trajectory data point from a given LMDB file and its
        corresponding index.

        Parameters
        ----------
        lmdb_index : int
            Index corresponding to which LMDB file to read from
        subindex : int
            Index corresponding to the data to retrieve from an LMDB file

        Returns
        -------
        Any
            Unpickled representation of the data
        """
        data = utils.get_data_from_index(lmdb_index, subindex, self._envs)
        return data

    @property
    @cache
    def keys(self) -> List[Tuple[int, int]]:
        return self._load_keys()

    def __getitem__(self, index: int) -> DataDict:
        """
        Implements the __getitem__ method that PyTorch `DataLoader` need
        to retrieve a piece of data. This implementation should not require
        tampering: child classes should just call `super().__getitem__(idx)`
        to get the raw data out, and post-process as required.

        The overall work flow is to look up the `keys` (a 2-tuple of
        LMDB file and subindex) from the `index` (i.e. `range(len(dataset))`),
        and return the unpickled data.

        Parameters
        ----------
        index : int
            Dataset index

        Returns
        -------
        Any
            Returns un-pickled data from the LMDB file.
        """
        keys = self.index_to_key(index)
        data = self.data_from_key(*keys)
        data["dataset"] = self.__class__.__name__
        # if some callable transforms have been provided, transform
        # the data sequentially
        if self.transforms:
            for transform in self.transforms:
                data = transform(data)
        return data

    def __len__(self) -> int:
        """
        This is a simple implementation so that the `__len__` function
        shouldn't need to be re-implemented for every dataset.
        """
        return len(self.keys)

    def __del__(self) -> None:
        """Teardown on exit to ensure all LMDB files are closed."""
        for env in self._envs:
            env.close()

    @staticmethod
    def collate_fn(batch: List[DataDict]) -> BatchDict:
        return utils.concatenate_keys(batch)

    def sample(self, num_samples: int) -> List[Any]:
        """
        Produce a set of random samples from this dataset.

        Samples _without_ replacement, and is intended for obtaining statistics
        about the dataset without iterating over its entirety.

        Parameters
        ----------
        num_samples : int
            Number of samples to draw

        Returns
        -------
        List[Any]
            List of samples from the dataset.
        """
        assert num_samples < len(self)
        indices = sample(range(len(self)), num_samples)
        samples = [self.__getitem__(i) for i in indices]
        return samples

    @property
    @abstractmethod
    def target_keys(self) -> Dict[str, List[str]]:
        """
        Indicates what the expected keys are for targets.

        Primarily serves as a method of peeking into what the task will entail
        for the task modules to initialize output heads.

        Returns
        -------
        Dict[str, List[str]]
            Target keys, nested by task type
        """
        ...

    @property
    def representation(self) -> str:
        return self._representation

    @representation.setter
    def representation(self, value: str) -> None:
        value = value.lower()
        assert value in [
            "graph",
            "point_cloud",
        ], "Supported representations are 'graph' and 'point_cloud'."
        self._representation = value

    @property
    def pad_keys(self) -> List[str]:
        ...

    @pad_keys.setter
    @abstractmethod
    def pad_keys(self, keys: List[str]) -> None:
        ...

    @classmethod
    def from_devset(cls, transforms: Optional[List[Callable]] = None, **kwargs):
        """
        Instantiate an instance of this dataset conveniently from the builtin
        devset.

        This method should be usable by child classes, and additional kwargs
        can be passed to modify its behavior further than providing transforms.

        Parameters
        ----------
        transforms : Optional[List[Callable]], optional
            List of transforms, by default None
        """
        return cls(cls.__devset__, transforms, **kwargs)


class PointCloudDataset(BaseLMDBDataset):
    def __init__(
        self,
        lmdb_root_path: Union[str, Path],
        transforms: Optional[List[Callable[..., Any]]] = None,
        full_pairwise: bool = True,
    ) -> None:
        super().__init__(lmdb_root_path, transforms)
        self.full_pairwise = full_pairwise
        self.representation = "point_cloud"

    @staticmethod
    def choose_dst_nodes(size: int, full_pairwise: bool) -> Dict[str, torch.Tensor]:
        r"""
        Generate indices for nodes to construct a point cloud with. If ``full_pairwise``
        is ``True``, the point cloud will be symmetric with shape ``[max_size, max_size]``,
        otherwise a random number of neighbors (ranging from 1 to ``max_size``) will be
        used to select ``dst_nodes``, with the resulting shape being ``[max_size, num_neighbors]``.

        Parameters
        ----------
        size : int
            Number of particles in the full system
        full_pairwise : bool
            Toggles whether to pair all nodes with all other nodes. Setting to ``False``
            will help improve memory footprint.

        Returns
        -------
        Dict[str, torch.Tensor]
            Key/value pair of source and destination node indices. If ``full_pairwise``,
            then the two tensors are identical.
        """
        src_indices = torch.arange(size)
        if not full_pairwise:
            num_neighbors = torch.randint(1, size, (1,)).item()
            dst_indices = torch.randperm(size)[:num_neighbors].sort().values
        else:
            dst_indices = src_indices
        return {"src_nodes": src_indices, "dst_nodes": dst_indices}
