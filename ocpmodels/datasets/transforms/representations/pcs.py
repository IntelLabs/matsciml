from typing import List, Union, Optional
from abc import abstractmethod

from ocpmodels.common import DataDict, package_registry
from ocpmodels.common.types import DataDict, GraphTypes, AbstractGraph
from ocpmodels.datasets.transforms.representations import RepresentationTransform
from ocpmodels.datasets import utils

"""
Transforms that create point cloud representations from graphs.
"""

__all__ = ["GraphToPointCloudTransform"]


class GraphToPointCloudTransform(RepresentationTransform):
    def __init__(self, backend: str, atom_centered: bool = True) -> None:
        """
        _summary_

        Parameters
        ----------
        backend : str
            Either 'dgl' or 'pyg'; specifies that graph framework to
            represent structures
        atom_centered : bool, optional
            If True, creates atom-centered point clouds; by default True
        """
        super().__init__(backend=backend)
        self.atom_centered = atom_centered

    def prologue(self, data: DataDict) -> None:
        assert self._check_for_type(
            data, GraphTypes
        ), f"No graphs to transform into point clouds!"
        assert data["dataset"] in [
            "IS2REDataset",
            "S2EFDataset",
        ], f"Dataset not from OCP; this transform should only be applied to IS2RE/S2EF."
        return super().prologue(data)

    if package_registry["dgl"]:
        import dgl

        def _convert_dgl(self, g: dgl.DGLGraph, data: DataDict) -> None:
            import dgl

            assert isinstance(
                g, dgl.DGLGraph
            ), f"Expected DGL graph as input, but got {g} which is type {type(g)}"
            features = g.ndata["atomic_numbers"]
            pos = g.ndata["pos"]
            # compute atom-centered point clouds
            if self.atom_centered:
                features = utils.point_cloud_featurization(features, features, 100)
                pos = pos[None, :] - pos[:, None]
            data["pos"] = pos
            data["pc_features"] = features

    if package_registry["pyg"]:
        import torch_geometric
        from torch_geometric.data import Data as PyGData

        @staticmethod
        def _convert_pyg(g, data: DataDict) -> None:
            ...

    def convert(self, data: DataDict) -> None:
        graph: AbstractGraph = data["graph"]
        if self.backend == "dgl":
            self._convert_dgl(graph, data)
        else:
            self._convert_pyg(graph, data)

    def epilogue(self, data: DataDict) -> None:
        try:
            del data["graph"]
        except KeyError:
            pass
        return super().prologue(data)
