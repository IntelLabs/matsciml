from abc import abstractmethod
from typing import Dict, List, Union

import torch
import dgl


class AbstractGraphTransform(object):
    @abstractmethod
    def __call__(
        self, data: Dict[str, Union[torch.Tensor, dgl.DGLGraph]]
    ) -> Dict[str, Union[torch.Tensor, dgl.DGLGraph]]:
        """
        Call function for an abstract graph transform.

        This abstract method defines the expected input and outputs;
        we retrieve a dictionary of data, and we expect a dictionary
        of data to come back out.

        In some sense, this might not be what you would think
        of canonically as a "transform" in that it's not operating
        in place, but rather as modular components of the pipeline.

        Parameters
        ----------
        data : Dict[str, Union[torch.Tensor, dgl.DGLGraph]]
            Item from an abstract dataset

        Returns
        -------
        Dict[str, Union[torch.Tensor, dgl.DGLGraph]]
            Transformed item from an abstract dataset
        """
        raise NotImplementedError


class DistancesTransform(AbstractGraphTransform):
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

    def __call__(
        self, data: Dict[str, Union[torch.Tensor, dgl.DGLGraph]]
    ) -> Dict[str, Union[torch.Tensor, dgl.DGLGraph]]:
        graph = data.get("graph")
        graph.apply_edges(self.pairwise_distance)
        return data


class GraphVariablesTransform(AbstractGraphTransform):
    """
    Transform to compute graph-level variables for use in models
    like MegNet. These will be included in the output dictionary
    as the "graph_variables" key.
    """

    def __call__(
        self, data: Dict[str, Union[torch.Tensor, dgl.DGLGraph]]
    ) -> Dict[str, Union[torch.Tensor, dgl.DGLGraph]]:
        # retrieve the DGL graph
        graph = data.get("graph")
        # isolate the subtrate
        mol_mask = graph.ndata["tags"] == 2
        charge_data = self._get_atomic_charge(graph, mol_mask)
        dist_data = self._get_distance_features(graph, mol_mask)
        # stack the variables together into a single vector
        graph_variables = torch.nan_to_num(torch.stack([*charge_data, *dist_data]))
        data["graph_variables"] = graph_variables
        return data

    @staticmethod
    def _get_atomic_charge(
        graph: dgl.DGLGraph, mol_mask: torch.Tensor
    ) -> List[torch.Tensor]:
        # extract out nodes that belong to the molecule
        surf_mask = ~mol_mask
        output = []
        for mask in [mol_mask, surf_mask]:
            atomic_numbers = graph.ndata["atomic_numbers"][mask]
            output.extend([atomic_numbers.mean(), atomic_numbers.std()])
        return output

    @staticmethod
    def _get_distance_features(
        graph: dgl.DGLGraph, mol_mask: torch.Tensor
    ) -> List[torch.Tensor]:
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
            atom_numbers = graph.ndata["atomic_numbers"][mask].unsqueeze(-1)
            positions = graph.ndata["pos"][mask]
            num_sum = atom_numbers.sum()
            coms.append((atom_numbers * positions).sum(dim=0) / num_sum)
        # compute distance between the center of masses
        sub_distance = torch.sqrt(((coms[0] - coms[1]) ** 2.0).sum())
        # compute the bond length/reduced mass statistics
        avg_dist, std_dist = graph.edata["r"].mean(), graph.edata["r"].std()
        avg_mu, std_mu = graph.edata["mu"].mean(), graph.edata["mu"].std()
        return [avg_dist, std_dist, avg_mu, std_mu, sub_distance]


class RemoveTagZeroNodes(AbstractGraphTransform):
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

    def __call__(
        self, data: Dict[str, Union[torch.Tensor, dgl.DGLGraph]]
    ) -> Dict[str, Union[torch.Tensor, dgl.DGLGraph]]:
        graph = data.get("graph")
        tag_zero_mask = graph.ndata["tags"] != 0
        select_node_indices = graph.nodes()[tag_zero_mask]
        # induce subgraph based on atom tags
        subgraph = dgl.node_subgraph(graph, select_node_indices)
        target_key = "graph" if self.overwrite else "subgraph"
        data[target_key] = subgraph
        return data


class GraphSupernode(AbstractGraphTransform):
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
        return self.atom_max_embed_index + 1

    def __call__(
        self, data: Dict[str, Union[torch.Tensor, dgl.DGLGraph]]
    ) -> Dict[str, Union[torch.Tensor, dgl.DGLGraph]]:
        graph = data.get("graph")
        # check to make sure the nodes
        assert (
            len(graph.ndata["tags"] == 0) > 0
        ), "No nodes with tag == 0 in `graph`! Please apply this transform before removing tag zero nodes."
        tag_zero_indices = graph.nodes()[graph.ndata["tags"] == 0]
        supernode_data = {
            "tags": torch.as_tensor([3], dtype=graph.ndata["tags"].dtype),
            "atomic_numbers": torch.as_tensor(
                [self.supernode_index], dtype=graph.ndata["atomic_numbers"].dtype
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
