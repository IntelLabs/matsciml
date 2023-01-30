# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: MIT License

from typing import Union, List, Any, Tuple, Callable, Optional, Dict
from pathlib import Path
from abc import abstractstaticmethod, abstractproperty
import functools
import pickle

import lmdb
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import dgl
from dgl.nn.pytorch.factory import KNNGraph
from munch import Munch


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


class BaseOCPDataset(Dataset):
    """
    Main purpose of this class is to inherit LMDB file
    reading.
    """

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

    @abstractproperty
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
        env = self._envs[lmdb_index]
        with env.begin() as txn:
            return pickle.loads(txn.get(f"{subindex}".encode("ascii")))

    @property
    @cache
    def keys(self) -> List[Tuple[int, int]]:
        return self._load_keys()

    def __getitem__(self, index: int) -> Any:
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
        # if some callable transforms have been provided, transform
        # the data sequentially
        if self.transforms:
            # TODO transform interface should act on a dictionary
            for transform in self.transforms:
                data = transform(data)
        return data

    # @cache
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

    @abstractstaticmethod
    def collate_fn(batch: List[Any]) -> List[Any]:
        raise NotImplementedError(
            "Collate function is not implemented for this class, {self.__class__.__name__}."
        )


class DGLDataset(BaseOCPDataset):
    @staticmethod
    def collate_fn(
        batch: List[Dict[str, Union[torch.Tensor, dgl.DGLGraph]]]
    ) -> Dict[str, Union[torch.Tensor, dgl.DGLGraph]]:
        """
        Collate a batch of DGL data together.

        A batch of DGL data comprises multiple keys with Tensor values,
        except for `graph` which contains a `DGLGraph`. For the former,
        we just batch them as one would with regular tensors, and for the
        latter, we use the native `dgl.batch` function to pack them together.

        Parameters
        ----------
        batch : List[Dict[str, Union[torch.Tensor, dgl.DGLGraph]]]
            A list containing individual IS2RE data points

        Returns
        -------
        Dict[str, Union[torch.Tensor, dgl.DGLGraph]]
            Dictionary with keys: ["graph", "natoms", "y", "sid", "fid", "cell"]
            of batched data.
        """
        batched_graphs = dgl.batch([entry["graph"] for entry in batch])
        batched_data = {"graph": batched_graphs}
        # get keys from the first batch entry
        keys = filter(lambda x: x != "graph", batch[0].keys())
        for key in keys:
            data = [entry.get(key) for entry in batch]
            if isinstance(data[0], torch.Tensor):
                data = torch.stack(data)
            else:
                data = torch.Tensor(data)
            batched_data[key] = data
        return batched_data

    @property
    def data_loader(self) -> dgl.dataloading.GraphDataLoader:
        """
        Return the bogstandard DGL graph dataloader.

        Honestly not sure if it functionally makes a difference from
        the stock PyTorch DataLoader, but for the sake of consistency
        here we are :P

        Returns
        -------
        dgl.dataloading.GraphDataLoader
            Referenece to the DGL DataLoader
        """
        return dgl.dataloading.GraphDataLoader


class PointCloudDataset(Dataset):
    """
    TODO reimplement using BaseOCPDataset as parent

    For better abstraction and performance, it would be worth looking
    into using BaseOCPDataset (i.e. straight from the LMDB without
    DGLGraphs) instead of these subclasses.

    Alternatively, this could be implemented as a transform instead,
    although then the collate function abstraction would break.
    """

    def __init__(
        self,
        dataset: DGLDataset,
        point_cloud_size: Optional[int] = 24,
        sample_size: Optional[int] = 80,
        transforms: Optional[List[Callable]] = None,
        natom_types: int = 100,
    ) -> None:
        super().__init__()
        # additional point cloud arguments needed for the KNN graph
        # and sampling
        self._pc_size = point_cloud_size
        self._sample_size = sample_size
        self._dataset = dataset
        self._natom_types = natom_types
        # construct a KNNGraph object for use
        self._knn = KNNGraph(self._pc_size)

    def __len__(self) -> int:
        return len(self._dataset)

    @property
    def data_loader(self) -> DataLoader:
        """
        Since this class just uses tensors, we are able to just use
        the regular PyTorch DataLoader class.

        Returns
        -------
        DataLoader
            Reference to the PyTorch DataLoader class
        """
        return DataLoader

    @property
    @cache
    def eye(self) -> torch.Tensor:
        """
        Creates and caches a diagonal tensor of dimensionality
        `natom_types`, representing a one-hot embedding for each
        atom type in our dataset.

        Returns
        -------
        torch.Tensor
            2D diagonal [N, N] tensor
        """
        return torch.eye(self._natom_types)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """
        This function retrieves a data point from the basis dataset
        (i.e. S2EF or IS2RE), and formats the data from the DGLGraph
        into a point cloud representation.

        We first construct a graph from k-NN based on the graph node
        positions, then randomly sampling connected nodes to extract
        a point cloud. Once this is completed, we copy over all of
        the additional data (i.e. labels/targets) from the original
        data.

        Parameters
        ----------
        index : int
            Index of the data point to retrieve; passed to the basis
            dataset.

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary of tensors containing the point cloud data
            and labels/targets.
        """
        # this retrieves a piece of data from the underlying S2EF or IS2RE
        # dataset; it's all the same anyway :D
        data = self._dataset.__getitem__(index)
        graph = data.get("graph")
        # compute point cloud from KNN - TODO try without KNN once
        knn_graph = graph  # self._knn(graph.ndata["pos"])
        (u, v) = knn_graph.edges()
        # # refer to the original graph to slice out data
        tags, nodes = graph.ndata["tags"], graph.nodes()
        molecule_nodes = [tags == 2]
        surface_nodes = [tags == 1]
        substrate_nodes = [tags == 0]
        molecule_idx = nodes[molecule_nodes]
        surface_idx = nodes[surface_nodes]
        substrate_idx = nodes[substrate_nodes]
        mol_idx = graph.nodes()[molecule_idx].tolist()
        surface_idx = graph.nodes()[surface_idx].tolist()
        substrate_idx = graph.nodes()[substrate_idx]
        # calculate the number of things we _need_ so we sample on top of that
        mol_surf_num = len(mol_idx) + len(surface_idx)
        # random sampling
        num_items = self._sample_size  # min(self._sample_size, len(substrate_idx))
        choose_items = max(num_items - mol_surf_num, 0)

        substrate_idx_sample_idx = substrate_idx[
            torch.randperm(min(choose_items, len(substrate_idx)))
        ]
        # currently not used, but TODO adapt to KNN sampling
        total_num = mol_surf_num + len(substrate_idx_sample_idx)
        substrate_idx_sample_idx = substrate_idx_sample_idx.tolist()
        substrate_indices = torch.LongTensor(
            mol_idx + surface_idx + substrate_idx_sample_idx
        )
        # get data and append to their respective lists
        source_types = graph.ndata["atomic_numbers"][mol_idx].long()
        dest_types = graph.ndata["atomic_numbers"][substrate_indices].long()
        # this represents the one-hot embedding lookup, relies on a cached
        # diagonal tensor
        source_onehot = self.eye[source_types][:, None]
        dest_onehot = self.eye[dest_types][None, :]
        plus = source_onehot + dest_onehot
        minus = source_onehot - dest_onehot
        # point cloud features as symmetric one-hot encodings
        # shape should be [natom_centers, neighbors, natom_types * 2]
        pc_features = torch.concat([plus, minus], axis=-1)
        # shift coordinates according to their atom centers
        # shape should be [natom_centers, neighbors, 3]
        pc_pos = (
            graph.ndata["pos"][substrate_indices][None, :]
            - graph.ndata["pos"][mol_idx][:, None]
        )

        # now we start getting the data out
        output_data = {
            "pc_features": pc_features,
            "pos": pc_pos,
            "sizes": len(substrate_idx),  # the size of the point cloud
            "nneighbors": len(dest_types),
            "ncenters": len(source_types),
        }
        if "force" in graph.ndata.keys():
            output_data["force"] = graph.ndata["force"][substrate_indices].squeeze()
        # copy over labels as well
        for key, value in data.items():
            if "graph" not in key:
                output_data[key] = value
        return output_data

    @staticmethod
    def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Collate function for the point cloud data representation.
        While there is osentisbly nothing special about the data (i.e. they
        are all just tensors), there is some padding required to match
        the maximum number of features - like a sequence.
        The way this is implemented should also be agnostic to the
        original dataset; the labels should be copied over correctly
        regardless of whether you're using S2EF or IS2RE as a basis.
        The expected shape of `pc_features` is: [N, natom_centers, nneighbors, natom_types * 2]
        In this iteration, we use `pad_sequence` to pad out the `natom_centers` and
        `nneighbors`, which may be variable for both.
        Parameters
        ----------
        batch : List[Dict[str, torch.Tensor]]
            List of individual data points
        Returns
        -------
        Dict[str, torch.Tensor]
            A dictionary containing the same keys as inputs, but
            each tensor is batched.
        """
        # assume keys are the same for each entry
        keys = batch[0].keys()
        output_dict = {}
        pad_keys = ["pos", "pc_features"]
        if "force" in keys:
            pad_keys.append("force")
        # for everything else, there's Mastercard
        for key in keys:
            if key not in pad_keys:
                data = [b[key] for b in batch]
                # stack tensors, otherwise just make a flat 1D tensor
                if isinstance(data[0], torch.Tensor):
                    result = torch.stack(data)
                elif isinstance(data[0], (dgl.DGLGraph, dgl.DGLHeteroGraph)):
                    result = dgl.batch(data)
                else:
                    result = torch.as_tensor(data)
                output_dict[key] = result
        # these keys are special because we have to pad to match the number of point clouds
        max_centers, max_neighbors = (
            output_dict["ncenters"].max().item(),
            output_dict["nneighbors"].max().item(),
        )
        batch_size = len(batch)
        for key in pad_keys:
            # force doesn't need to be padded
            if key != "force":
                # get the last dimension of the feature
                example = batch[0][key]
                feat_dim = example.size(-1)
                # preallocate zeros tensor to hold everything
                batched_data = torch.zeros(
                    (batch_size, max_centers, max_neighbors, feat_dim),
                    dtype=example.dtype,
                )
                # iterate over samples, and copy of data to zero-padded tensors
                for index, sample in enumerate(batch):
                    lengths = sample[key].shape
                    batched_data[
                        index, : lengths[0], : lengths[1], : lengths[2]
                    ] = sample[key][:, :, :]
                output_dict[key] = batched_data
        return output_dict
