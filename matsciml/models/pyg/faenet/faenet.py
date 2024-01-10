"""
FAENet: Frame Averaging Equivariant graph neural Network
Simple, scalable and expressive model for property prediction on 3D atomic systems.
"""
from __future__ import annotations

from copy import deepcopy

import torch
from einops import reduce
from torch import nn
from torch.nn import Linear

from matsciml.common.registry import registry
from matsciml.common.types import AbstractGraph, BatchDict, DataDict, Embeddings
from matsciml.common.utils import radius_graph_pbc
from matsciml.models.base import AbstractPyGModel
from matsciml.models.pyg.faenet.helper import *
from matsciml.models.pyg.faenet.layers import *


@registry.register_model("FAENet")
class FAENet(AbstractPyGModel):
    r"""Non-symmetry preserving GNN model for 3D atomic systems,
    called FAENet: Frame Averaging Equivariant Network.

    Args:
        cutoff (float): Cutoff distance for interatomic interactions.
            (default: :obj:`6.0`)
        preprocess (callable): Pre-processing function for the data. This function
            should accept a data object as input and return a tuple containing the following:
            atomic numbers, batch indices, final adjacency, relative positions, pairwise distances.
            Examples of valid preprocessing functions include `pbc_preprocess`,
            `base_preprocess`, or custom functions.
        act (str): Activation function
            (default: `silu`)
        max_num_neighbors (int): The maximum number of neighbors to
            collect for each node within the :attr:`cutoff` distance.
            (default: `40`)
        hidden_channels (int): Hidden embedding size.
            (default: `128`)
        tag_hidden_channels (int): Hidden tag embedding size.
            (default: :obj:`32`)
        pg_hidden_channels (int): Hidden period and group embedding size.
            (default: :obj:`32`)
        phys_embeds (bool): Do we include fixed physics-aware embeddings.
            (default: :obj: `True`)
        phys_hidden_channels (int): Hidden size of learnable physics-aware embeddings.
            (default: :obj:`0`)
        num_interactions (int): The number of interaction (i.e. message passing) blocks.
            (default: :obj:`4`)
        num_gaussians (int): The number of gaussians :math:`\mu` to encode distance info.
            (default: :obj:`50`)
        num_filters (int): The size of convolutional filters.
            (default: :obj:`128`)
        second_layer_MLP (bool): Use 2-layers MLP at the end of the Embedding block.
            (default: :obj:`False`)
        skip_co (str): Add a skip connection between each interaction block and
            energy-head. (`False`, `"add"`, `"concat"`, `"concat_atom"`)
        mp_type (str): Specificies the Message Passing type of the interaction block.
            (`"base"`, `"updownscale_base"`, `"updownscale"`, `"updown_local_env"`, `"simple"`):
        graph_norm (bool): Whether to apply batch norm after every linear layer.
            (default: :obj:`True`)
        complex_mp (bool); Whether to add a second layer MLP at the end of each Interaction
            (default: :obj:`True`)
        energy_head (str): Method to compute energy prediction
            from atom representations.
            (`None`, `"weighted-av-initial-embeds"`, `"weighted-av-final-embeds"`)
        out_dim (int): size of the output tensor for graph-level predicted properties ("energy")
            Allows to predict multiple properties at the same time.
            (default: :obj:`1`)
        pred_as_dict (bool): Set to False to return a (property) prediction tensor.
            By default, predictions are returned as a dictionary with several keys (e.g. energy, forces)
            (default: :obj:`True`)
        regress_forces (str): Specifies if we predict forces or not, and how
            do we predict them. (`None` or `""`, `"direct"`, `"direct_with_gradient_target"`)
        force_decoder_type (str): Specifies the type of force decoder
            (`"simple"`, `"mlp"`, `"res"`, `"res_updown"`)
        force_decoder_model_config (dict): contains information about the
            for decoder architecture (e.g. number of layers, hidden size).
        frame_averaging (str): Transform method *already* used.
            Can be 2D FA, 3D FA, Data Augmentation or no FA, respectively denoted by
            (`"2D"`, `"3D"`, `"DA"`, `""`)
        crystal_task (bool, optional): Whether crystals (molecules) are considered.
            If they are, the unit cell (3x3) is affected by frame averaged and expected as attribute.
            (default: :obj:`True`)
    """

    def __init__(
        self,
        cutoff: float = 6.0,
        preprocess: str | callable = "pbc_preprocess",
        act: str = "silu",
        max_num_neighbors: int = 40,
        hidden_channels: int = 128,
        tag_hidden_channels: int = 32,
        pg_hidden_channels: int = 32,
        phys_embeds: bool = True,
        phys_hidden_channels: int = 0,
        num_interactions: int = 4,
        num_gaussians: int = 50,
        num_filters: int = 128,
        second_layer_MLP: bool = True,
        skip_co: str = "concat",
        mp_type: str = "updownscale_base",
        graph_norm: bool = True,
        complex_mp: bool = False,
        energy_head: str | None = None,
        out_dim: int = 1,
        pred_as_dict: bool = True,
        regress_forces: str | None = None,
        force_decoder_type: str | None = "mlp",
        force_decoder_model_config: dict | None = {"hidden_channels": 128},
        embedding_size: int = 100,
        average_frame_embeddings: bool = False,
        frame_averaging: str = "3D",
        crystal_task: bool = True,
        **kwargs,
    ):
        super().__init__(atom_embedding_dim=118)

        self.act = act
        self.complex_mp = complex_mp
        self.cutoff = cutoff
        self.energy_head = energy_head
        self.force_decoder_type = force_decoder_type
        self.force_decoder_model_config = force_decoder_model_config
        self.graph_norm = graph_norm
        self.hidden_channels = hidden_channels
        self.max_num_neighbors = max_num_neighbors
        self.mp_type = mp_type
        self.num_filters = num_filters
        self.num_gaussians = num_gaussians
        self.num_interactions = num_interactions
        self.pg_hidden_channels = pg_hidden_channels
        self.phys_embeds = phys_embeds
        self.phys_hidden_channels = phys_hidden_channels
        self.regress_forces = regress_forces
        self.second_layer_MLP = second_layer_MLP
        self.skip_co = skip_co
        self.tag_hidden_channels = tag_hidden_channels
        self.preprocess = preprocess
        self.pred_as_dict = pred_as_dict
        self.emb_size = embedding_size
        self.average_frame_embeddings = average_frame_embeddings
        self.frame_averaging = frame_averaging
        self.crystal_task = crystal_task

        if isinstance(self.preprocess, str):
            self.preprocess = eval(self.preprocess)

        if not isinstance(self.regress_forces, str):
            assert self.regress_forces is False or self.regress_forces is None, (
                "regress_forces must be a string "
                + "('', 'direct', 'direct_with_gradient_target') or False or None"
            )
            self.regress_forces = ""

        if self.mp_type == "simple":
            self.num_filters = self.hidden_channels

        self.act = (
            getattr(nn.functional, self.act) if isinstance(self.act, str) else self.act
        )
        assert callable(self.act), (
            "act must be a callable function or a string "
            + "describing that function in torch.nn.functional"
        )

        # Gaussian Basis
        self.distance_expansion = GaussianSmearing(0.0, self.cutoff, self.num_gaussians)

        # Embedding block
        self.embed_block = EmbeddingBlock(
            self.num_gaussians,
            self.num_filters,
            self.hidden_channels,
            self.tag_hidden_channels,
            self.pg_hidden_channels,
            self.phys_hidden_channels,
            self.phys_embeds,
            self.act,
            self.second_layer_MLP,
            self.emb_size,
        )
        self.atom_embedding = self.embed_block

        # Interaction block
        self.interaction_blocks = nn.ModuleList(
            [
                InteractionBlock(
                    self.hidden_channels,
                    self.num_filters,
                    self.act,
                    self.mp_type,
                    self.complex_mp,
                    self.graph_norm,
                )
                for _ in range(self.num_interactions)
            ],
        )

        # Output block
        self.output_block = OutputBlock(
            self.energy_head,
            self.hidden_channels,
            self.act,
            out_dim,
        )

        # Energy head
        if self.energy_head == "weighted-av-initial-embeds":
            self.w_lin = Linear(self.hidden_channels, 1)

        # Skip co
        if self.skip_co == "concat":
            self.mlp_skip_co = Linear(out_dim * (self.num_interactions + 1), out_dim)
        elif self.skip_co == "concat_atom":
            self.mlp_skip_co = Linear(
                ((self.num_interactions + 1) * self.hidden_channels),
                self.hidden_channels,
            )

    def get_embed_inputs(
        self,
        data,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z, batch, edge_index, rel_pos, edge_weight = self.preprocess(
            data,
            self.cutoff,
            self.max_num_neighbors,
        )
        return z, batch, edge_index, rel_pos, edge_weight

    def read_batch(self, batch: BatchDict) -> DataDict:
        r"""
        Extracts data needed by FAENet from the batch and graph
        structures.

        Node features
        Edge features

        Parameters
        ----------
        batch : BatchDict
            Batch of data to be processed

        Returns
        -------
        DataDict
            Input data for FAENet as a dictionary.
        """

        data = {"graph": batch.get("graph")}
        graph = batch.get("graph")
        for key in ["edge_feats", "graph_feats"]:
            data[key] = getattr(graph, key, None)
        pos: torch.Tensor = getattr(graph, "pos")
        data["pos"] = pos
        data["graph"].cell = batch["cell"]
        data["graph"].natoms = batch["natoms"].squeeze(-1).to(torch.int32)
        edge_index, cell_offsets, neighbors = radius_graph_pbc(
            data["graph"],
            self.cutoff,
            50,
        )
        data["graph"].edge_index = edge_index
        data["graph"].cell_offsets = cell_offsets
        data["graph"].neighbors = neighbors
        atomic_numbers, batch, edge_index, rel_pos, edge_weight = self.get_embed_inputs(
            data["graph"],
        )
        edge_attr = self.distance_expansion(edge_weight)  # RBF of pairwise distances
        node_embeddings = self.atom_embedding(atomic_numbers, rel_pos, edge_attr)
        # optionally can fuse into a single tensor with `self.join_position_embeddings`
        data["node_feats"] = node_embeddings
        return data

    def energy_forward(self, data, preproc: bool = True) -> Embeddings:
        """Predicts any graph-level property (e.g. energy) for 3D atomic systems.

        Args:
            data (data.Batch): Batch of graphs data objects.
            preproc (bool): Whether to apply (any given) preprocessing to the graph.
                Default to True.

        Returns:
            (dict): predicted properties for each graph (key: "energy")
                and final atomic representations (key: "hidden_state")
        """
        # Pre-process data (e.g. pbc, cutoff graph, etc.)
        # Should output all necessary attributes, in correct format.
        z, batch, edge_index, rel_pos, edge_weight = self.get_embed_inputs(data)
        edge_attr = self.distance_expansion(edge_weight)  # RBF of pairwise distances
        assert z.dim() == 1 and z.dtype == torch.long

        # Embedding block
        h, e = self.embed_block(
            z,
            rel_pos,
            edge_attr,
            data.tags if hasattr(data, "tags") else None,
        )

        # Compute atom weights for late energy head
        if self.energy_head == "weighted-av-initial-embeds":
            alpha = self.w_lin(h)
        else:
            alpha = None

        # Interaction blocks
        energy_skip_co = []
        for interaction in self.interaction_blocks:
            if self.skip_co == "concat_atom":
                energy_skip_co.append(h)
            elif self.skip_co:
                energy_skip_co.append(
                    self.output_block(h, edge_index, edge_weight, batch, alpha),
                )
            h = h + interaction(h, edge_index, e)

        # Atom skip-co
        if self.skip_co == "concat_atom":
            energy_skip_co.append(h)
            node_embedding = self.act(
                self.mlp_skip_co(torch.cat(energy_skip_co, dim=1)),
            )
        else:
            node_embedding = h
        graph_embedding = self.output_block(
            node_embedding,
            edge_index,
            edge_weight,
            batch,
            alpha,
        )
        return Embeddings(graph_embedding, node_embedding)

    def first_forward(
        self,
        graph: AbstractGraph,
        **kwargs,
    ) -> Embeddings:
        """
        Actually flowing data through architecture. First we predict
        the energy, and optionally gradients.

        Args:
            data (Data): input data object, with 3D atom positions (pos)
            mode (str): train or inference mode
            preproc (bool): Whether to preprocess (pbc, cutoff graph)
                the input graph or point cloud. Default: True.

        Returns:
            (dict): predicted energy, forces and final atomic hidden states
        """
        if self.training:
            mode = "train"
        else:
            mode = "inference"
        preproc = True
        data = graph

        # energy gradient w.r.t. positions will be computed
        if mode == "train" or self.regress_forces == "from_energy":
            data.pos.requires_grad_(True)

        # produce final embeddings after going through model
        embeddings = self.energy_forward(data, preproc)
        return embeddings

    def _forward(
        self,
        graph: AbstractGraph,
        node_feats: torch.Tensor,
        pos: torch.Tensor,
        edge_feats: torch.Tensor | None = None,
        **kwargs,
    ) -> Embeddings:
        """Perform a model forward pass when frame averaging is applied.

        Parameters
        ----------
        graph : dgl.DGLGraph
            A single or batch of molecular graphs

        Parameters
        ----------
        graph : AbstractGraph
            Instance of a PyG graph data structure
        node_feats : torch.Tensor
            Atomic embeddings obtained from nn.Embedding
        pos : torch.Tensor
            XYZ coordinates of each atom
        edge_feats : Optional[torch.Tensor], optional
            Tensor containing interatomic distances, by default None and unused.
        graph_feats : Optional[torch.Tensor], optional
            Graph-based properties, by default None and unused.

        Returns
        -------
        Embeddings
            Data structure containing node and graph level embeddings.
            Node embeddings correspond to after the node projection layer.
        """

        batch = graph

        if not hasattr(batch, "natoms"):
            batch.natoms = torch.unique(batch.batch, return_counts=True)[1]

        # check that frame averaging properties are available
        for key in ["fa_pos", "fa_cell", "fa_rot"]:
            if not hasattr(batch, key):
                raise KeyError(
                    f"Graph is expected to have property {key}: include frame averaging transform!",
                )

        # Distinguish Frame Averaging prediction from traditional case.
        if self.frame_averaging and self.frame_averaging != "DA":
            original_pos = batch.pos
            original_cell = getattr(batch, "cell", None)

            # Compute model prediction for each frame
            all_embeddings = []
            for frame_idx, frame in enumerate(batch.fa_pos):
                # set positions to current frame
                batch.pos = frame
                if self.crystal_task:
                    batch.cell = batch.fa_cell[frame_idx]
                # Forward pass
                embeddings = self.first_forward(batch)
                all_embeddings.append(embeddings)
            batch.pos = original_pos
            batch.cell = original_cell
            # now stack up embeddings into a single tensor
            node_embeddings = torch.stack(
                [frame.point_embedding for frame in all_embeddings],
                dim=1,
            )
            graph_embeddings = torch.stack(
                [frame.system_embedding for frame in all_embeddings],
                dim=1,
            )
            # if we're averaging the frame embeddings directly
            if self.average_frame_embeddings:
                node_embeddings = reduce(
                    node_embeddings,
                    "b f h -> b h",
                    reduction="mean",
                )
                graph_embeddings = reduce(
                    graph_embeddings,
                    "b f h -> b h",
                    reduction="mean",
                )
            all_embeddings = Embeddings(graph_embeddings, node_embeddings)

        # Traditional case (no frame averaging)
        else:
            all_embeddings = self.first_forward(deepcopy(batch))
        return all_embeddings

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())
