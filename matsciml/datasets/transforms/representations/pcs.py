from __future__ import annotations

from functools import partial
from typing import List, Optional, Union

import torch

from matsciml.common import DataDict, package_registry
from matsciml.common.types import AbstractGraph, DataDict, GraphTypes
from matsciml.datasets import utils
from matsciml.datasets.base import BaseLMDBDataset
from matsciml.datasets.transforms.representations import RepresentationTransform

"""
Transforms that create point cloud representations from graphs.
"""

if package_registry["dgl"]:
    import dgl
    from dgl import DGLGraph

if package_registry["pyg"]:
    import torch_geometric
    from torch_geometric.data import Data as PyGGraph

__all__ = ["GraphToPointCloudTransform", "OCPGraphToPointCloudTransform"]


class GraphToPointCloudTransform(RepresentationTransform):
    def __init__(self, backend: str, full_pairwise: bool = True) -> None:
        r"""
        Convert a graph data sample into a point cloud.

        The ``full_pairwise`` argument toggles between a complete node-node pairwise
        data representation and a randomly sampled "end point" format, which
        aims to knock down the memory requirements of a point cloud.

        In the case of ``full_pairwise`` being ``True``, then the point cloud
        will be symmetric (shape [N, N, D] for feature dimension D). If it's
        ``False``, then the destination (``dim=1``) shape will be M, where M
        can be any number between 1 and N, the size of the point cloud (shape [N, M, D]).

        In this current implementation, the atom positions are left as is and
        left up to the neural network model (via the ``read_batch``) method
        to construct the positions in the right way to enable autograd for forces.

        The keys we return as part of the data sample are:

        - ``pos`` [N, 3]
        - ``src_nodes``, ``dst_nodes``, N/N(M) as node indices
        - ``pc_features`` [N, N(M), 200] as node features
        - ``sizes`` scalar, number of particles in the whole system (N)

        Parameters
        ----------
        backend : str
            Either 'dgl' or 'pyg'; specifies that graph framework to
            represent structures
        full_pairwise : bool, optional
            If True, every node is compared against every other node; by default True.
            If False, we randomly sample ``dst`` nodes to construct the point cloud.
        """
        super().__init__(backend=backend)
        self.full_pairwise = full_pairwise

    def setup_transform(self, dataset: BaseLMDBDataset) -> None:
        """
        This modifies the dataset's collate function by replacing it with
        a partial with pad_keys specified.

        Parameters
        ----------
        dataset : BaseLMDBDataset
            A dataset object which is a subclass of `BaseLMDBDataset`.
        """
        dataset.representation = "point_cloud"
        # we will pack point cloud features, but not positions
        collate_fn = partial(
            utils.concatenate_keys,
            pad_keys=["pc_features"],
            unpacked_keys=["sizes", "src_nodes", "dst_nodes"],
        )
        dataset.collate_fn = staticmethod(collate_fn).__func__
        return super().setup_transform(dataset)

    def prologue(self, data: DataDict) -> None:
        assert self._check_for_type(
            data,
            GraphTypes,
        ), f"No graphs to transform into point clouds!"
        assert data["dataset"] in [
            "IS2REDataset",
            "S2EFDataset",
        ], f"Dataset not from OCP; this transform should only be applied to IS2RE/S2EF."
        return super().prologue(data)

    if package_registry["dgl"]:

        def _convert_dgl(self, g: dgl.DGLGraph, data: DataDict) -> None:
            assert isinstance(
                g,
                dgl.DGLGraph,
            ), f"Expected DGL graph as input, but got {g} which is type {type(g)}"
            features = g.ndata["atomic_numbers"].long()
            system_size = len(features)
            src_indices = torch.arange(system_size)
            if not self.full_pairwise:
                num_neighbors = torch.randint(1, system_size, (1,)).item()
                # extract out a random number of neighbors and sort the indices
                dst_indices = torch.randperm(system_size)[:num_neighbors].sort().values
            else:
                dst_indices = src_indices
            pos = g.ndata["pos"]
            # extract out point cloud features
            features = utils.point_cloud_featurization(
                features[src_indices],
                features[dst_indices],
                100,
            )
            data["pos"] = pos  # left as N, 3
            data["pc_features"] = features
            data["sizes"] = system_size
            data["src_nodes"] = src_indices
            data["dst_nodes"] = dst_indices

    if package_registry["pyg"]:

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


class OCPGraphToPointCloudTransform(GraphToPointCloudTransform):
    def __init__(
        self,
        backend: str,
        sample_size: int = 5,
        full_pairwise: bool = True,
    ) -> None:
        r"""
        Convert a graph data sample into a point cloud, with additional semantics
        for dealing with the Open Catalyst dataset which has labelled nodes.

        The ``full_pairwise`` behaves slightly differently from the base class:
        we use the full node set (molecule + substrate + surface nodes) as N,
        and the molecule node set as M. If ``full_pairwise`` is ``True``, the
        shape is [N, N, :], otherwise [M, N, :]. In contrast to the base class,
        N >= M, which allows for a more compact representation if not pairwise.

        In this current implementation, the atom positions are left as is and
        left up to the neural network model (via the ``read_batch``) method
        to construct the positions in the right way to enable autograd for forces.

        The keys we return as part of the data sample are:

        - ``pos`` [N, 3]
        - ``src_nodes``, ``dst_nodes``, N(M)/N as node indices
        - ``pc_features`` [N(M), N, 200] as node features
        - ``sizes`` scalar, the larger number of particles in the whole system (N)

        Parameters
        ----------
        backend : str
            Either 'dgl' or 'pyg'; specifies that graph framework to
            represent structures
        full_pairwise : bool, optional
            If True, every node is compared against every other node; by default True.
            If False, we randomly sample ``dst`` nodes to construct the point cloud.
        """
        super().__init__(backend, full_pairwise)
        self.sample_size = sample_size

    @staticmethod
    def _extract_indices(tags: torch.Tensor, nodes: torch.Tensor) -> list[list[int]]:
        """
        Extract out indices of nodes, given a tensor containing the OCP
        tags. This is written as in a framework agnostic way, with the
        intention to be re-used between PyG and DGL workflows.

        Parameters
        ----------
        tags : torch.Tensor
            1D Long tensor containing 0,1,2 for each node corresponding
            to molecule, surface, and substrate nodes respectively
        nodes : torch.Tensor
            1D tensor of node IDs

        Returns
        -------
        List[List[int]]
            List of three lists, corresponding to node indices for
            each category of separated nodes
        """
        molecule_idx = nodes[[tags == 2]]
        surface_idx = nodes[[tags == 1]]
        substrate_idx = nodes[[tags == 0]]
        # get nodes out
        molecule_nodes = nodes[molecule_idx].tolist()
        surface_nodes = nodes[surface_idx].tolist()
        substrate_nodes = nodes[substrate_idx].tolist()
        return molecule_nodes, surface_nodes, substrate_nodes

    def _pick_src_dst(
        self,
        molecule_nodes: list[int],
        surface_nodes: list[int],
        substrate_nodes: list[int],
    ) -> list[list[int]]:
        """
        Separates nodes into source and destination, as part of creating a more
        compact, molecule/atom centered representation of the point cloud. For
        each atom that is marked as part of the adsorbate/molecule, our positions
        and featurization factors in pairwise interactions with other molecule atoms
        as well as a random sampling of surface/substrate atoms.

        Parameters
        ----------
        molecule_nodes : List[int]
            List of indices corresponding to nodes that constitute the adsorbate
        surface_nodes : List[int]
            List of indices corresponding to nodes that constitute the surface
        substrate_nodes : List[int]
            List of indices corresponding to nodes that constitute the substrate

        Returns
        -------
        src_nodes, dst_nodes
            List of node indices used for source/destination designation
        """
        num_samples = max(
            self.sample_size - len(molecule_nodes) + len(surface_nodes),
            0,
        )
        if isinstance(substrate_nodes, list):
            substrate_nodes = torch.tensor(substrate_nodes)
        neighbor_idx = substrate_nodes[
            torch.randperm(min(num_samples, len(substrate_nodes)))
        ].tolist()
        src_nodes = torch.LongTensor(molecule_nodes)
        dst_nodes = torch.LongTensor(molecule_nodes + surface_nodes + neighbor_idx)
        # in the full pairwise, make the point cloud neighbors symmetric
        if self.full_pairwise:
            src_nodes = dst_nodes
        return src_nodes, dst_nodes

    if package_registry["dgl"]:

        def _convert_dgl(self, g: DGLGraph, data: DataDict) -> None:
            tags, nodes, atomic_numbers = (
                g.ndata["tags"],
                g.nodes(),
                g.ndata["atomic_numbers"],
            )
            # extract out nodes based on tags, then separate into src/dst point cloud
            # neighborhoods
            molecule_nodes, surface_nodes, substrate_nodes = self._extract_indices(
                tags,
                nodes,
            )
            # the pairwise logic is located inside `_pick_src_dst`; in the affirmative
            # case, src_nodes == dst_nodes
            src_nodes, dst_nodes = self._pick_src_dst(
                molecule_nodes,
                surface_nodes,
                substrate_nodes,
            )
            # create point cloud featurizations
            src_features = atomic_numbers[src_nodes].long()
            dst_features = atomic_numbers[dst_nodes].long()
            pc_features = utils.point_cloud_featurization(
                src_features,
                dst_features,
                max_types=100,
            )
            # node positions still kept as N, 3
            node_pos = g.ndata["pos"]
            # copy data over to dictionary
            data["pc_features"] = pc_features
            data["pos"] = node_pos
            data["src_nodes"] = src_nodes
            data["dst_nodes"] = dst_nodes
            # we retain the full set of nodes for indexing, so the size
            # of the full pos tensor is different from other datasets
            data["sizes"] = len(node_pos)
            data["force"] = g.ndata["force"][dst_nodes].squeeze()
