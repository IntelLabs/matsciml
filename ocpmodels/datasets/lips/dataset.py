from importlib.util import find_spec
from typing import Tuple, Dict, List, Union, Any, Optional, Callable
from pathlib import Path

import torch
import numpy as np

from ocpmodels.datasets.base import BaseLMDBDataset


_has_dgl = find_spec("dgl") is not None
_has_pyg = find_spec("torch_geometric") is not None


def item_from_structure(data: Any, *keys: str) -> Any:
    """
    Function to recurse through an object and retrieve a nested attribute.

    Parameters
    ----------
    data : Any
        Basically any Python object
    keys : str
        Any variable number of keys to recurse through.

    Returns
    -------
    Any
        Retrieved nested attribute/object

    Raises
    ------
    KeyError
        If a key is not present, raise KeyError.
    """
    for key in keys:
        assert key in data.keys(), f"{key} not found in {data}."
        if isinstance(data, dict):
            data = data.get(key)
        else:
            data = getattr(data, key)
    return data


class LiPSDataset(BaseLMDBDataset):
    def index_to_key(self, index: int) -> Tuple[int]:
        return (0, index)

    @staticmethod
    def collate_fn(
        batch: List[Dict[str, Union[torch.Tensor, float]]]
    ) -> Dict[str, Union[torch.Tensor, float]]:
        joint_data = {}
        sample = batch[0]
        pad_keys = ["pos", "atomic_numbers", "force"]
        # get the biggest point cloud size for padding
        if any([key in sample.keys() for key in pad_keys]):
            max_size = max([s["pos"].size(0) for s in batch])
            batch_size = len(batch)
        for key, value in sample.items():
            # for dictionaries, we need to go one level deeper
            if isinstance(value, dict):
                if key != "target_keys":
                    joint_data[key] = {}
                    for subkey, subvalue in value.items():
                        data = [item_from_structure(s, key, subkey) for s in batch]
                        # for numeric types, cast to a float tensor
                        if isinstance(subvalue, (int, float)):
                            data = torch.FloatTensor(data).unsqueeze(-1)
                        elif isinstance(subvalue, torch.Tensor):
                            data = torch.vstack(data)
                        # for string types, we just return a list
                        else:
                            pass
                        joint_data[key][subkey] = data
            elif key in pad_keys:
                assert isinstance(
                    value, torch.Tensor
                ), f"{key} in batch should be a tensor."
                # get the dimensionality
                tensor_dim = value.size(-1)
                if value.ndim == 2:
                    data = torch.zeros(
                        batch_size, max_size, tensor_dim, dtype=value.dtype
                    )
                else:
                    data = torch.zeros(batch_size, max_size, dtype=value.dtype)
                # pack the batched tensor with each sample now
                for index, s in enumerate(batch):
                    tensor: torch.Tensor = s[key]
                    if tensor.ndim == 2:
                        lengths = tensor.shape
                        data[index, : lengths[0], : lengths[1]] = tensor[:, :]
                    else:
                        length = len(tensor)
                        data[index, :length] = tensor[:]
                joint_data[key] = data
            else:
                # aggregate all the data
                data = [s.get(key) for s in batch]
                if isinstance(value, torch.Tensor) and key != "distance_matrix":
                    data = torch.vstack(data)
                elif isinstance(value, (int, float)):
                    data = torch.FloatTensor(data).unsqueeze(-1)
                # return anything else as just a list
                joint_data[key] = data
        joint_data["target_types"] = sample["target_types"]
        # make sure we are pointing to the same object to help with memory
        for key, value in joint_data["targets"].items():
            joint_data[key] = value
        return joint_data

    def data_from_key(
        self, lmdb_index: int, subindex: int
    ) -> Dict[str, Union[float, torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Retrieve a sample from the LMDB file.

        The only modification here from the base method is to tack on some
        metadata about expected targets: for this dataset, we have energy
        and force labels.

        Parameters
        ----------
        lmdb_index : int
            Index of the LMDB file
        subindex : int
            Index of the sample within the LMDB file

        Returns
        -------
        Dict[str, Union[float, torch.Tensor, Dict[str, torch.Tensor]]]
            A single sample from this dataset
        """
        data = super().data_from_key(lmdb_index, subindex)
        data["targets"] = {}
        data["target_types"] = {"regression": [], "classification": []}
        for key in ["energy", "force"]:
            data["targets"][key] = data.get(key)
            data["target_types"]["regression"].append(key)
        return data


if _has_dgl:
    import dgl

    class DGLLiPSDataset(LiPSDataset):
        def __init__(self, lmdb_root_path: Union[str, Path], cutoff_dist: float = 5., transforms: Optional[List[Callable]] = None) -> None:
            super().__init__(lmdb_root_path, transforms)
            self.cutoff_dist = cutoff_dist

        def data_from_key(self, lmdb_index: int, subindex: int) -> Dict[str, Union[float, torch.Tensor, Dict[str, torch.Tensor]]]:
            data = super().data_from_key(lmdb_index, subindex)
            pos: torch.Tensor = data["pos"]
            dist_mat = torch.cdist(pos, pos, p=2).numpy()
            lower_tri = np.tril(dist_mat)
            # mask out self loops and atoms that are too far away
            mask = (0.0 < lower_tri) * (lower_tri < self.cutoff_dist)
            adj_list = np.argwhere(mask).tolist()  # DGLGraph only takes lists
            # number of nodes has to be passed explicitly since cutoff
            # radius may result in shorter adj_list
            graph = dgl.graph(adj_list, num_nodes=len(data["atomic_numbers"]))
            for key in ["pos", "atomic_numbers"]:
                graph.ndata[key] = data.get(key)
            data["graph"] = graph
            return data

        @staticmethod
        def collate_fn(
            batch: List[Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]]
        ) -> Dict[str, Union[torch.Tensor, dgl.DGLGraph, Dict[str, torch.Tensor]]]:
            """
            Collate function for DGLGraph variant of the Materials Project.

            Basically uses the same workflow as that for `MaterialsProjectDataset`,
            but with the added step of calling `dgl.batch` on the graph data
            that is left unprocessed by the parent method.

            Parameters
            ----------
            batch : List[Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]]
                List of Materials Project samples

            Returns
            -------
            Dict[str, Union[torch.Tensor, dgl.DGLGraph, Dict[str, torch.Tensor]]]
                Batched data, including graph
            """
            batched_data = super(
                DGLLiPSDataset, DGLLiPSDataset
            ).collate_fn(batch)
            batched_data["graph"] = dgl.batch(batched_data["graph"])
            return batched_data
