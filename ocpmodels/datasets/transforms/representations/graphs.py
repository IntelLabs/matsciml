from typing import List, Union, Optional
from logging import getLogger

import torch
import numpy as np

from ocpmodels.common import DataDict, package_registry
from ocpmodels.common.types import DataDict, GraphTypes, AbstractGraph
from ocpmodels.datasets.transforms.representations import RepresentationTransform

"""
Construct graphs from point clouds
"""

__all__ = ["PointCloudToGraphTransform"]

if package_registry["dgl"]:
    from dgl import DGLGraph
    from dgl import graph as dgl_graph

if package_registry["pyg"]:
    import torch_geometric
    from torch_geometric.data import Data as PyGGraph

log = getLogger(__name__)


class PointCloudToGraphTransform(RepresentationTransform):
    def __init__(
        self,
        backend: str,
        cutoff_dist: float = 7.0,
        node_keys: List[str] = ["coords", "atomic_numbers", "force"],
        edge_keys: Optional[List[str]] = None,
    ) -> None:
        super().__init__(backend=backend)
        self.cutoff_dist = cutoff_dist
        self.node_keys = node_keys
        self.edge_keys = edge_keys

    @property
    def node_keys(self) -> List[str]:
        return self._node_keys

    @node_keys.setter
    def node_keys(self, values: List[str]) -> None:
        values = set(values)
        for key in ["coords", "atomic_numbers"]:
            values.add(key)
        self._node_keys = list(values)

    def prologue(self, data: DataDict) -> None:
        assert not self._check_for_type(
            data, GraphTypes
        ), "Data structure already contains a graph: transform shouldn't be required."
        # check for keys needed to construct the graph
        for key in ["coords", "atomic_numbers"]:
            assert key in data, f"Expected {key} in sample. Found: {list(data.keys())}"
        return super().prologue(data)

    @staticmethod
    def node_distances(coords: torch.Tensor) -> torch.Tensor:
        assert coords.ndim == 2, "Expected atom coordinates to be 2D tensor."
        assert coords.size(-1) == 3, "Expected XYZ coordinates."
        return torch.cdist(coords, coords, p=2).to(torch.float16)

    @staticmethod
    def edges_from_dist(
        dist_mat: Union[np.ndarray, torch.Tensor], cutoff: float
    ) -> List[List[int]]:
        if isinstance(dist_mat, np.ndarray):
            dist_mat = torch.from_numpy(dist_mat)
        lower_tri = torch.tril(dist_mat)
        # mask out self loops and atoms that are too far away
        mask = (0.0 < lower_tri) * (lower_tri < cutoff)
        adj_list = torch.argwhere(mask).tolist()
        return adj_list

    if package_registry["dgl"]:

        def _copy_node_keys_dgl(self, data: DataDict, graph: DGLGraph) -> None:
            # DGL variant of node data copying
            for key in self.node_keys:
                try:
                    graph.ndata[key] = data[key]
                except KeyError:
                    log.warning(
                        f"Expected node data '{key}' but was not found in data sample: {list(data.keys())}"
                    )
            graph.ndata["pos"] = graph.ndata["coords"]

        def _copy_edge_keys_dgl(self, data: DataDict, graph: DGLGraph) -> None:
            # DGL variant of edge data copying
            edge_keys = getattr(self, "edge_keys", None)
            if edge_keys:
                for key in edge_keys:
                    try:
                        graph.edata[key] = data[key]
                    except KeyError:
                        log.warning(
                            f"Expected edge data {key} but was not found in data sample: {list(data.keys())}"
                        )

        def _convert_dgl(self, data: DataDict) -> None:
            atom_numbers = data["atomic_numbers"]
            coords = data["coords"]
            num_nodes = len(atom_numbers)
            # skip edge calculation if the distance matrix
            # exists already
            if "distance_matrix" not in data:
                dist_mat = self.node_distances(coords)
            else:
                dist_mat = data.get("distance_matrix")
            adj_list = self.edges_from_dist(dist_mat, self.cutoff_dist)
            g = dgl_graph(adj_list, num_nodes=num_nodes)
            data["graph"] = g

    if package_registry["pyg"]:

        def _copy_node_keys_pyg(self, data: DataDict, graph: PyGGraph) -> None:
            for key in self.node_keys:
                try:
                    setattr(graph, key, data[key])
                except KeyError:
                    log.warning(
                        f"Expected node data '{key}' but was not found in data sample: {list(data.keys())}"
                    )

        def _copy_edge_keys_pyg(self, data: DataDict, graph: PyGGraph) -> None:
            edge_keys = getattr(self, "edge_keys", None)
            if edge_keys:
                for key in edge_keys:
                    try:
                        setattr(graph, key, data[key])
                    except KeyError:
                        log.warning(
                            f"Expected edge data '{key}' but was not found in data sample: {list(data.keys())}"
                        )

        def _convert_pyg(self, data: DataDict) -> None:
            """
            Structure data into a PyG format, which at the minimum, packs the
            atomic numbers and nuclear coordinates. Mutates the dictionary
            of data inplace.

            Parameters
            ----------
            data : DataDict
                Data structure read from base class
            """
            atom_numbers = data["atomic_numbers"]
            coords = data["coords"]
            if "distance_matrix" not in data:
                dist_mat = self.node_distances(coords)
            else:
                dist_mat = data.get("distance_matrix")
            # convert ensure edges are in the right format for PyG
            edge_index = torch.LongTensor(
                self.edges_from_dist(dist_mat, self.cutoff_dist)
            )
            # if not in the expected shape, transpose and reformat layout
            if edge_index.size(0) != 2 and edge_index.size(1) == 2:
                edge_index = edge_index.T.contiguous()
            g = PyGGraph(edge_index=edge_index, pos=coords)
            g.atomic_numbers = atom_numbers
            data["graph"] = g

    def convert(self, data: DataDict) -> None:
        if self.backend == "dgl":
            self._convert_dgl(data)
        else:
            self._convert_pyg(data)

    def copy_node_keys(self, data: DataDict, graph: AbstractGraph) -> None:
        if self.backend == "dgl":
            self._copy_node_keys_dgl(data, graph)
        else:
            self._copy_node_keys_pyg(data, graph)

    def copy_edge_keys(self, data: DataDict, graph: AbstractGraph) -> None:
        if self.backend == "dgl":
            self._copy_edge_keys_dgl(data, graph)
        else:
            self._copy_edge_keys_pyg(data, graph)

    def epilogue(self, data: DataDict) -> None:
        g = data["graph"]
        # copy over data to graph structure
        self.copy_node_keys(data, g)
        self.copy_edge_keys(data, g)
        # remove unused/redundant keys in DataDict
        for key in [
            "coords",
            "pos",
            "pc_features",
            "distance_matrix",
            "atomic_numbers",
        ]:
            try:
                del data[key]
            except KeyError:
                pass
        return super().prologue(data)
