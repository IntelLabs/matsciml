from __future__ import annotations

from abc import abstractmethod
from typing import Any, Dict, List, Optional, Union

import torch
from pymatgen.core import Lattice

from matsciml.common import DataDict, package_registry
from matsciml.common.types import DataDict
from matsciml.datasets.transforms.base import AbstractDataTransform

if package_registry["dgl"]:
    import dgl

if package_registry["pyg"]:
    import torch_geometric

__all__ = [
    "DistancesTransform",
    "GraphVariablesTransform",
    "RemoveTagZeroNodes",
    "AtomicSuperNodes",
    "GraphSuperNodes",
    "GraphReordering",
    "CoordinateScaling",
    "COMShift",
    "ScaleRegressionTargets",
    "UnitCellCalculator",
    "DummyTransform",
]


class DistancesTransform(AbstractDataTransform):
    """
    Compute the distance and reduced mass between a pair of
    bonded nodes.

    Returns these as "r" and "mu" keys on edges.
    """

    @staticmethod
    def pairwise_distance(edges):
        r = ((edges.src["pos"] - edges.dst["pos"]) ** 2.0).sum(
            [
                1,
            ],
            keepdim=True,
        ) ** 0.5
        m_a, m_b = edges.src["atomic_numbers"], edges.dst["atomic_numbers"]
        mu = (m_a * m_b) / (m_a + m_b)
        return {"r": r, "mu": mu}

    def __call__(self, data: DataDict) -> DataDict:
        graph = data.get("graph")
        graph.apply_edges(self.pairwise_distance)
        return data


class GraphVariablesTransform(AbstractDataTransform):
    """
    Transform to compute graph-level variables for use in models
    like MegNet. These will be included in the output dictionary
    as the "graph_variables" key.
    """

    def __call__(self, data: DataDict) -> DataDict:
        # retrieve the DGL graph
        graph = data.get("graph")
        # isolate the subtrate
        if "tags" in graph.ndata:
            mol_mask = graph.ndata["tags"] == 2
        else:
            mol_mask = torch.ones(graph.num_nodes(), dtype=bool)
        charge_data = self._get_atomic_charge(graph, mol_mask)
        dist_data = self._get_distance_features(graph, mol_mask)
        # stack the variables together into a single vector
        graph_variables = torch.nan_to_num(torch.stack([*charge_data, *dist_data]))
        data["graph_variables"] = graph_variables
        return data

    @staticmethod
    def _get_atomic_charge(
        graph: dgl.DGLGraph,
        mol_mask: torch.Tensor,
    ) -> list[torch.Tensor]:
        # extract out nodes that belong to the molecule
        surf_mask = ~mol_mask
        output = []
        for mask in [mol_mask, surf_mask]:
            atomic_numbers = graph.ndata["atomic_numbers"][mask].float()
            output.extend([atomic_numbers.mean(), atomic_numbers.std()])
        return output

    @staticmethod
    def _get_distance_features(
        graph: dgl.DGLGraph,
        mol_mask: torch.Tensor,
    ) -> list[torch.Tensor]:
        """
        Compute spatial features for the graph level variables.
        This computes two summary features: the average bond length
        between every pair, and the distance between the substrate's
        and the surface's center of masses.

        Parameters
        ----------
        graph : dgl.DGLGraph
            Input DGL graph containing the molecule/surface data
        mol_mask : torch.Tensor
            Boolean mask, with `True` elements corresponding to
            nodes that belong to the substrate

        Returns
        -------
        List[torch.Tensor]
            5-element list containing distance based features
            aggregated for the whole graph
        """
        assert (
            "r" in graph.edata
        ), f"'r' key missing from edge data; please include `DistancesTransform` before this transform."
        surf_mask = ~mol_mask
        # center of masses
        coms = []
        for mask in [mol_mask, surf_mask]:
            atom_numbers = graph.ndata["atomic_numbers"][mask].unsqueeze(-1).float()
            positions = graph.ndata["pos"][mask]
            num_sum = atom_numbers.sum()
            coms.append((atom_numbers * positions).sum(dim=0) / num_sum)
        # compute distance between the center of masses
        sub_distance = torch.sqrt(((coms[0] - coms[1]) ** 2.0).sum())
        # compute the bond length/reduced mass statistics
        avg_dist, std_dist = graph.edata["r"].mean(), graph.edata["r"].std()
        avg_mu, std_mu = graph.edata["mu"].mean(), graph.edata["mu"].std()
        return [avg_dist, std_dist, avg_mu, std_mu, sub_distance]


class RemoveTagZeroNodes(AbstractDataTransform):
    """
    Create a subgraph representation where atoms tagged with
    zero are absent.

    The `overwrite` kwarg, which is by default `True`, will overwrite
    the `graph` key in a batch to avoid duplication of data. If set
    to `False`, the new subgraph can be referenced with the `subgraph`
    key.

    TODO make sure the properties of the graph are updated, but a glance
    at the original implementation the `dgl.node_subgraph` should take
    care of everything.
    """

    def __init__(self, overwrite: bool = True) -> None:
        super().__init__()
        self.overwrite = overwrite

    def __call__(self, data: DataDict) -> DataDict:
        graph = data.get("graph")
        tag_zero_mask = graph.ndata["tags"] != 0
        select_node_indices = graph.nodes()[tag_zero_mask]
        # induce subgraph based on atom tags
        subgraph = dgl.node_subgraph(graph, select_node_indices)
        target_key = "graph" if self.overwrite else "subgraph"
        data[target_key] = subgraph
        # need to update force targets as the number of nodes will be fewer
        if "force" in subgraph.ndata and self.overwrite:
            data["targets"]["force"] = subgraph.ndata["force"]
        return data


class GraphSuperNodes(AbstractDataTransform):
    """
    Generates a super node based on tag zero nodes.

    This is intended to be run _prior_ to `RemoveTagZeroNodes`, as it
    does require aggregating data over all of the tag zero nodes.
    """

    def __init__(self, atom_max_embed_index: int) -> None:
        super().__init__()
        self.atom_max_embed_index = atom_max_embed_index

    @property
    def supernode_index(self) -> int:
        # mostly for the sake of abstraction; this just makes it easier to
        # understand the embedding lookup index for the graph supernode
        return self.atom_max_embed_index

    def __call__(self, data: DataDict) -> DataDict:
        graph = data.get("graph")
        # check to make sure the nodes
        assert (
            len(graph.ndata["tags"] == 0) > 0
        ), "No nodes with tag == 0 in `graph`! Please apply this transform before removing tag zero nodes."
        tag_zero_indices = graph.nodes()[graph.ndata["tags"] == 0]
        supernode_data = {
            "tags": torch.as_tensor([3], dtype=graph.ndata["tags"].dtype),
            "atomic_numbers": torch.as_tensor(
                [self.supernode_index],
                dtype=graph.ndata["atomic_numbers"].dtype,
            ),
            "fixed": torch.as_tensor([1], dtype=graph.ndata["fixed"].dtype),
        }
        # extract out the tag zero nodes to compute averages
        tag_zero_subgraph = dgl.node_subgraph(graph, tag_zero_indices)
        # for 2D tensors, like positions and forces, get the average value
        for key, tensor in tag_zero_subgraph.ndata.items():
            if tensor.ndim == 2:
                # have to unsqueeze to make the node appending work
                supernode_data[key] = tensor.mean(0).unsqueeze(0)
        new_graph = dgl.add_nodes(graph, 1, data=supernode_data)
        supernode_index = new_graph.num_nodes() - 1
        # add edges to every existing node with the super node, except itself
        new_graph = dgl.add_edges(
            new_graph,
            u=[supernode_index for _ in range(supernode_index)],
            v=[index for index in range(supernode_index)],
        )
        # overwrite the existing graph
        data["graph"] = new_graph
        return data


class AtomicSuperNodes(AbstractDataTransform):
    def __init__(self, atom_max_embedding: int) -> None:
        super().__init__()
        self.atom_max_embedding = atom_max_embedding

    def super_atom_embedding(self, atomic_number: int, graph_super_node: bool) -> int:
        """
        This function shifts the super node indexes from the individual atom
        embedding indices. Depending on whether or not a graph super node
        was included, we will also shift the index accordingly.

        An example is with 100 atom types, the maximum superatom value
        would be 199; if the graph supernode embedding is included as
        well, then it would be 200 (with embedding index 100 being the
        graph supernode value.)

        Parameters
        ----------
        atomic_number : int
            Atomic number index to offset

        Returns
        -------
        int
            Offset value of the superatom embedding
        """
        index = self.atom_max_embedding + atomic_number
        if graph_super_node:
            index += 1
        return index

    def __call__(self, data: DataDict) -> DataDict:
        graph = data.get("graph")
        # check to see if there are graph supernodes already, which will offset
        # the atomic supernode indices
        if torch.any(graph.ndata["tags"] == 3):
            graph_super_node = True
        else:
            graph_super_node = False
        # extract the tag zero data again
        tag_zero_mask = graph.ndata["tags"] == 0
        assert len(
            tag_zero_mask > 0,
        ), "No nodes with tag == 0 in `graph`! Please apply this transform before removing tag zero nodes."
        tag_zero_indices = graph.nodes()[tag_zero_mask]
        tag_zero_subgraph = dgl.node_subgraph(graph, tag_zero_indices)
        # slice out atomic numbers that are only in the sub-surface
        unique_atomic_numbers = tag_zero_subgraph.ndata["atomic_numbers"].unique()
        # get the edges
        u, v = graph.edges()
        for atomic_number in unique_atomic_numbers:
            # slice out the nodes that match the element and is in the subsurface
            all_nodes_mask = (graph.ndata["tags"] == 0) * (
                graph.ndata["atomic_numbers"] == atomic_number
            )
            # similar to the other supernode featurization, but different in the tag
            # and in the way the embedding number is computed
            supernode_data = {
                "tags": torch.as_tensor([4], dtype=graph.ndata["tags"].dtype),
                "atomic_numbers": torch.as_tensor(
                    [self.super_atom_embedding(atomic_number, graph_super_node)],
                    dtype=graph.ndata["atomic_numbers"].dtype,
                ),
                "fixed": torch.as_tensor([1], dtype=graph.ndata["fixed"].dtype),
            }
            for key, tensor in graph.ndata.items():
                masked_tensor = tensor[all_nodes_mask]
                if masked_tensor.ndim == 2:
                    # have to unsqueeze to make the node appending work
                    supernode_data[key] = masked_tensor.mean(0).unsqueeze(0)
            # now we need to build the edges
            node_indices = graph.nodes()[all_nodes_mask]
            # rebuild a list of nodes to connect with
            src_targets, dst_targets = (
                u[torch.isin(u, node_indices)],
                v[torch.isin(v, node_indices)],
            )
            # add edges to each node only once
            joint = torch.cat([src_targets, dst_targets]).unique()
            num_edges = len(joint)
            graph = dgl.add_nodes(graph, 1, data=supernode_data)
            # add edges joining the supernode to existing connections
            new_node_index = graph.num_nodes() - 1
            graph = dgl.add_edges(
                graph,
                [new_node_index for _ in range(num_edges)],
                joint,
            )
        data["graph"] = graph
        return data


class GraphReordering(AbstractDataTransform):
    """
    Implements a graph node/edge reordering transform, which nominally
    may help with graph processing performance. This transform
    wraps `dgl.reorder_graph`, and the `init` method allows the user to
    customize how the graphs are ordered.
    """

    def __init__(
        self,
        node_algo: str | None = None,
        edge_algo: str = "src",
        **sort_kwargs,
    ) -> None:
        super().__init__()
        self.node_algo = node_algo
        self.edge_algo = edge_algo
        self.sort_kwargs = sort_kwargs

    def __call__(self, data: DataDict) -> DataDict:
        graph = data.get("graph")
        graph = dgl.reorder_graph(
            graph,
            self.node_algo,
            self.edge_algo,
            permute_config=self.sort_kwargs,
        )
        data["graph"] = graph
        return data


class CoordinateScaling(AbstractDataTransform):
    def __init__(self, value: float, key: str = "pos") -> None:
        self.value = value
        self.key = key

    def __call__(self, data: DataDict) -> DataDict:
        if "graph" in data:
            graph = data["graph"]
            assert self.key in graph.ndata, f"{self.key} not found in graph node data."
            target_tensor: torch.Tensor = data["graph"].ndata[self.key]
        elif self.key in data:
            target_tensor: torch.Tensor = data[self.key]
        else:
            raise KeyError(
                f"{self.key} not found in samples, or graph was not present.",
            )
        # perform inplace rescaling
        target_tensor.mul_(self.value)
        return data


class COMShift(AbstractDataTransform):
    """
    Shift coordinates of a point cloud or graph based on its center of mass.

    For convenience, the default is to just use atomic numbers (since they're
    proportional to mass). The center of mass is then subtracted off the atom
    positions.
    """

    def __init__(self, pos_key: str = "pos", mass_key: str = "atomic_numbers") -> None:
        self.pos_key = pos_key
        self.mass_key = mass_key

    def __call__(self, data: DataDict) -> DataDict:
        if "graph" in data:
            target_dict = data["graph"].ndata
        else:
            target_dict = data
        assert (
            self.pos_key in target_dict
        ), f"Coordinates expected in {self.pos_key}, but not found in data."
        assert (
            self.mass_key in target_dict
        ), f"Masses expected in {self.mass_key}, but not found in data."
        coords: torch.Tensor = target_dict[self.pos_key]
        masses: torch.Tensor = target_dict[self.mass_key]
        mass_sum = masses.sum()
        # sum over nodes, mass times position
        com = torch.einsum("i,ij->j", masses.float(), coords) / mass_sum
        # make dimensions match up
        coords.sub_(com.unsqueeze(0))
        return data


class ScaleRegressionTargets(AbstractDataTransform):
    def __init__(
        self,
        value: float | None = None,
        values: dict[str, float] | None = None,
    ) -> None:
        if value is None and values is None:
            raise ValueError(
                f"No values provided - must provide either value or values arguments.",
            )
        self.value = value if value else 1.0
        self.values = values if values else {}

    def __call__(self, data: DataDict) -> DataDict:
        target_keys = data["target_types"]["regression"]
        for key in target_keys:
            # first check if it's been specified
            scaling = self.values.get(key, None)
            if not scaling:
                scaling = self.value
            data["targets"][key] *= scaling
        return data


class UnitCellCalculator(AbstractDataTransform):
    def __call__(self, data: DataDict) -> DataDict:
        lattice_features = data.get("lattice_features", None)
        if lattice_features is None:
            lattice_params = data.get("lattice_params", None)
        else:
            lattice_params = lattice_features.get("lattice_params", None)
        if lattice_params is None:
            raise KeyError(
                f"No lattice parameters available to calculate unit cell matrix.",
            )
        abc, angles = lattice_params[:3], lattice_params[3:]
        angles = torch.FloatTensor(
            tuple(angle * (180.0 / torch.pi) for angle in angles),
        )
        lattice_matrix = torch.Tensor(Lattice.from_parameters(*abc, *angles).matrix)
        data["cell"] = lattice_matrix.unsqueeze(0)
        return data


class DummyTransform(AbstractDataTransform):
    """
    Implements a dummy transform class for testing the behavior
    of transform pipelines.

    All this class does is leave a ``touched`` key with ``True``
    in an incoming data sample. This value can be checked to
    ensure transforms are working as intended, and to check
    whether preprocessed data has been serialized correctly.
    """

    def __call__(self, data: DataDict) -> DataDict:
        data["touched"] = True
        return data
