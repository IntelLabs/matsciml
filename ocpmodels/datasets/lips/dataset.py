from importlib.util import find_spec
from typing import Tuple, Dict, List, Union, Any, Optional, Callable
from pathlib import Path

import torch
import numpy as np
from ocpmodels.common.types import BatchDict, DataDict

from ocpmodels.datasets.base import BaseLMDBDataset
from ocpmodels.datasets.utils import concatenate_keys, point_cloud_featurization
from ocpmodels.common.registry import registry


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


@registry.register_dataset("LiPSDataset")
class LiPSDataset(BaseLMDBDataset):
    def index_to_key(self, index: int) -> Tuple[int]:
        return (0, index)

    @staticmethod
    def collate_fn(batch: List[DataDict]) -> BatchDict:
        return concatenate_keys(batch, pad_keys=["force", "pc_features", "pos"])

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

        coords = data["pos"]
        data["pos"] = coords[None, :] - coords[:, None]
        data["coords"] = coords
        atom_numbers = torch.LongTensor(data["atomic_numbers"])
        # uses one-hot encoding featurization
        pc_features = point_cloud_featurization(atom_numbers, atom_numbers, 200)
        # keep atomic numbers for graph featurization
        data["atomic_numbers"] = atom_numbers
        data["pc_features"] = pc_features
        data["num_particles"] = len(atom_numbers)

        data["targets"] = {}
        data["target_types"] = {"regression": [], "classification": []}
        for key in ["energy", "force"]:
            data["targets"][key] = data.get(key)
            data["target_types"]["regression"].append(key)
        return data

    @property
    def target_keys(self) -> Dict[str, List[str]]:
        return {"regression": ["energy", "force"]}


if _has_dgl:
    import dgl

    class DGLLiPSDataset(LiPSDataset):
        def __init__(
            self,
            lmdb_root_path: Union[str, Path],
            cutoff_dist: float = 5.0,
            transforms: Optional[List[Callable]] = None,
        ) -> None:
            super().__init__(lmdb_root_path, transforms)
            self.cutoff_dist = cutoff_dist

        def data_from_key(
            self, lmdb_index: int, subindex: int
        ) -> Dict[str, Union[float, torch.Tensor, Dict[str, torch.Tensor]]]:
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
            # make DGL graph symmetric
            data["graph"] = dgl.to_bidirected(graph, copy_ndata=True)
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
            batched_data = super(DGLLiPSDataset, DGLLiPSDataset).collate_fn(batch)
            batched_data["graph"] = dgl.batch(batched_data["graph"])
            return batched_data
