from __future__ import annotations

from typing import Tuple, Union

import pandas as pd
import torch
import torch.nn as nn
from mendeleev.fetch import fetch_ionization_energies, fetch_table
from torch import nn
from torch.nn import Embedding, Linear
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.norm import GraphNorm
from torch_scatter import scatter

from matsciml.models.pyg.faenet.helper import *


class PhysEmbedding(nn.Module):
    """
    Create physics-aware embeddings for each atom based their properties.

    Args:
        props (bool, optional): Create an embedding of physical
            properties. (default: :obj:`True`)
        props_grad (bool, optional): Learn a physics-aware embedding
            instead of keeping it fixed. (default: :obj:`False`)
        pg (bool, optional): Learn two embeddings based on period and
            group information respectively. (default: :obj:`False`)
        short (bool, optional): Remove all columns containing NaN values.
            (default: :obj:`False`)
    """

    def __init__(
        self,
        props: bool = True,
        props_grad: bool = False,
        pg: bool = False,
        short: bool = False,
        emb_size: int = 100,
    ) -> None:
        super().__init__()

        self.properties_list = [
            "atomic_radius",
            "atomic_volume",
            "density",
            "dipole_polarizability",
            "electron_affinity",
            "en_allen",
            "vdw_radius",
            "metallic_radius",
            "metallic_radius_c12",
            "covalent_radius_pyykko_double",
            "covalent_radius_pyykko_triple",
            "covalent_radius_pyykko",
            "IE1",
            "IE2",
        ]
        self.group_size = 0
        self.period_size = 0
        self.n_properties = 0

        self.props = props
        self.props_grad = props_grad
        self.pg = pg
        self.short = short

        self.emb_size = emb_size

        group = None
        period = None

        # Load table with all properties of all periodic table elements
        df = fetch_table("elements")
        df = df.set_index("atomic_number")

        # Add ionization energy
        ies = fetch_ionization_energies(degree=[1, 2])
        df = pd.concat([df, ies], axis=1)

        # Fetch group and period data
        if pg:
            df.group_id = df.group_id.fillna(value=19.0)
            # using fixed group size for embedding, was: df.group_id.unique().shape[0]
            self.group_size = self.emb_size
            group = torch.cat(
                [
                    torch.ones(1, dtype=torch.long),
                    torch.tensor(df.group_id.loc[:100].values, dtype=torch.long),
                ],
            )
            # using fixed period size, was: df.period.loc[:100].unique().shape[0]
            self.period_size = self.emb_size
            period = torch.cat(
                [
                    torch.ones(1, dtype=torch.long),
                    torch.tensor(df.period.loc[:100].values, dtype=torch.long),
                ],
            )

        self.register_buffer("group", group)
        self.register_buffer("period", period)

        # Create an embedding of physical properties
        if props:
            # Select only potentially relevant elements
            df = df[self.properties_list]
            df = df.loc[: self.emb_size, :]

            # Normalize
            df = (df - df.mean()) / df.std()
            # normalized_df=(df-df.min())/(df.max()-df.min())

            # Process 'NaN' values and remove further non-essential columns
            if self.short:
                self.properties_list = df.columns[~df.isnull().any()].tolist()
                df = df[self.properties_list]
            else:
                self.properties_list = df.columns[
                    pd.isnull(df).sum() < int(1 / 2 * df.shape[0])
                ].tolist()
                df = df[self.properties_list]
                col_missing_val = df.columns[df.isna().any()].tolist()
                df[col_missing_val] = df[col_missing_val].fillna(
                    value=df[col_missing_val].mean(),
                )

            self.n_properties = len(df.columns)
            properties = torch.cat(
                [
                    torch.zeros(1, self.n_properties),
                    torch.from_numpy(df.values).float(),
                ],
            )
            if props_grad:
                self.register_parameter("properties", nn.Parameter(properties))
            else:
                self.register_buffer("properties", properties)


class EmbeddingBlock(nn.Module):
    """Initialise atom and edge representations."""

    def __init__(
        self,
        num_gaussians: int,
        num_filters: int,
        hidden_channels: int,
        tag_hidden_channels: int,
        pg_hidden_channels: int,
        phys_hidden_channels: int,
        phys_embeds: int,
        act: callable,
        second_layer_MLP: bool,
        emb_size: int,
    ) -> None:
        super().__init__()
        self.act = act
        self.use_tag = tag_hidden_channels > 0
        self.use_pg = pg_hidden_channels > 0
        self.use_mlp_phys = phys_hidden_channels > 0 and phys_embeds
        self.second_layer_MLP = second_layer_MLP

        # --- Node embedding ---
        self.emb_size = emb_size

        # Phys embeddings
        self.phys_emb = PhysEmbedding(
            props=phys_embeds,
            props_grad=phys_hidden_channels > 0,
            pg=self.use_pg,
            emb_size=emb_size,
        )
        # With MLP
        if self.use_mlp_phys:
            self.phys_lin = Linear(self.phys_emb.n_properties, phys_hidden_channels)
        else:
            phys_hidden_channels = self.phys_emb.n_properties

        # Period + group embeddings
        if self.use_pg:
            self.period_embedding = Embedding(
                self.phys_emb.period_size,
                pg_hidden_channels,
                padding_idx=0,
            )
            self.group_embedding = Embedding(
                self.phys_emb.group_size,
                pg_hidden_channels,
                padding_idx=0,
            )

        # Tag embedding
        if tag_hidden_channels:
            self.tag_embedding = Embedding(3, tag_hidden_channels, padding_idx=0)

        # Main embedding
        self.emb = Embedding(
            self.emb_size,
            hidden_channels
            - tag_hidden_channels
            - phys_hidden_channels
            - 2 * pg_hidden_channels,
            padding_idx=0,
        )

        # MLP
        self.lin = Linear(hidden_channels, hidden_channels)
        if self.second_layer_MLP:
            self.lin_2 = Linear(hidden_channels, hidden_channels)

        # --- Edge embedding ---
        self.lin_e1 = Linear(3, num_filters // 2)  # r_ij
        self.lin_e12 = Linear(num_gaussians, num_filters - (num_filters // 2))  # d_ij

        if self.second_layer_MLP:
            self.lin_e2 = Linear(num_filters, num_filters)

        self.reset_parameters()

    def reset_parameters(self):
        self.emb.reset_parameters()
        if self.use_mlp_phys:
            nn.init.xavier_uniform_(self.phys_lin.weight)
        if self.use_tag:
            self.tag_embedding.reset_parameters()
        if self.use_pg:
            self.period_embedding.reset_parameters()
            self.group_embedding.reset_parameters()
        nn.init.xavier_uniform_(self.lin.weight)
        self.lin.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.lin_e1.weight)
        self.lin_e1.bias.data.fill_(0)
        if self.second_layer_MLP:
            nn.init.xavier_uniform_(self.lin_2.weight)
            self.lin_2.bias.data.fill_(0)
            nn.init.xavier_uniform_(self.lin_e2.weight)
            self.lin_e2.bias.data.fill_(0)

    def forward(
        self,
        z: torch.Tensor,
        rel_pos: torch.Tensor,
        edge_attr: torch.Tensor,
        tag: torch.Tensor | None = None,
        subnodes=None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the Embedding block.
        Called in FAENet to generate initial atom and edge representations.

        Args:
            z (tensor): atomic numbers. (num_atoms, )
            rel_pos (tensor): relative atomic positions. (num_edges, 3)
            edge_attr (tensor): RBF of pairwise distances. (num_edges, num_gaussians)
            tag (tensor, optional): atom information specific to OCP. Defaults to None.

        Returns:
            (tensor, tensor): atom embeddings, edge embeddings
        """

        # --- Edge embedding --
        rel_pos = self.lin_e1(rel_pos)  # r_ij
        edge_attr = self.lin_e12(edge_attr)  # d_ij
        e = torch.cat((rel_pos, edge_attr), dim=1)
        e = self.act(e)  # can comment out

        if self.second_layer_MLP:
            # e = self.lin_e2(e)
            e = self.act(self.lin_e2(e))

        # --- Node embedding --

        # Create atom embeddings based on its characteristic number
        h = self.emb(z)

        # if self.phys_emb.device != h.device:
        #     self.phys_emb = self.phys_emb.to(h.device)

        # Concat tag embedding
        if self.use_tag:
            h_tag = self.tag_embedding(tag)
            h = torch.cat((h, h_tag), dim=1)

        # Concat physics embeddings
        if self.phys_emb.n_properties > 0:
            h_phys = self.phys_emb.properties[z]
            if self.use_mlp_phys:
                h_phys = self.phys_lin(h_phys)
            h = torch.cat((h, h_phys), dim=1)

        # Concat period & group embedding
        if self.use_pg:
            h_period = self.period_embedding(self.phys_emb.period[z])
            h_group = self.group_embedding(self.phys_emb.group[z])
            h = torch.cat((h, h_period, h_group), dim=1)

        # MLP
        h = self.act(self.lin(h))
        if self.second_layer_MLP:
            h = self.act(self.lin_2(h))
        return h, e


class InteractionBlock(MessagePassing):
    """Updates atom representations through custom message passing."""

    def __init__(
        self,
        hidden_channels: int,
        num_filters: int,
        act: callable,
        mp_type: str,
        complex_mp: bool,
        graph_norm: bool,
    ) -> None:
        super().__init__()
        self.act = act
        self.mp_type = mp_type
        self.hidden_channels = hidden_channels
        self.complex_mp = complex_mp
        self.graph_norm = graph_norm
        if graph_norm:
            self.graph_norm = GraphNorm(
                hidden_channels if "updown" not in self.mp_type else num_filters,
            )

        if self.mp_type == "simple":
            self.lin_h = nn.Linear(hidden_channels, hidden_channels)

        elif self.mp_type == "updownscale":
            self.lin_geom = nn.Linear(num_filters, num_filters)
            self.lin_down = nn.Linear(hidden_channels, num_filters)
            self.lin_up = nn.Linear(num_filters, hidden_channels)

        elif self.mp_type == "updownscale_base":
            self.lin_geom = nn.Linear(num_filters + 2 * hidden_channels, num_filters)
            self.lin_down = nn.Linear(hidden_channels, num_filters)
            self.lin_up = nn.Linear(num_filters, hidden_channels)

        elif self.mp_type == "updown_local_env":
            self.lin_down = nn.Linear(hidden_channels, num_filters)
            self.lin_geom = nn.Linear(num_filters, num_filters)
            self.lin_up = nn.Linear(2 * num_filters, hidden_channels)

        else:  # base
            self.lin_geom = nn.Linear(
                num_filters + 2 * hidden_channels,
                hidden_channels,
            )
            self.lin_h = nn.Linear(hidden_channels, hidden_channels)

        if self.complex_mp:
            self.other_mlp = nn.Linear(hidden_channels, hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        if self.mp_type != "simple":
            nn.init.xavier_uniform_(self.lin_geom.weight)
            self.lin_geom.bias.data.fill_(0)
        if self.complex_mp:
            nn.init.xavier_uniform_(self.other_mlp.weight)
            self.other_mlp.bias.data.fill_(0)
        if self.mp_type in {"updownscale", "updownscale_base", "updown_local_env"}:
            nn.init.xavier_uniform_(self.lin_up.weight)
            self.lin_up.bias.data.fill_(0)
            nn.init.xavier_uniform_(self.lin_down.weight)
            self.lin_down.bias.data.fill_(0)
        else:
            nn.init.xavier_uniform_(self.lin_h.weight)
            self.lin_h.bias.data.fill_(0)

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        e: torch.Tesnor,
    ) -> torch.Tensor:
        """Forward pass of the Interaction block.
        Called in FAENet forward pass to update atom representations.

        Args:
            h (tensor): atom embedddings. (num_atoms, hidden_channels)
            edge_index (tensor): adjacency matrix. (2, num_edges)
            e (tensor): edge embeddings. (num_edges, num_filters)

        Returns:
            (tensor): updated atom embeddings
        """
        # Define edge embedding
        if self.mp_type in {"base", "updownscale_base"}:
            e = torch.cat([e, h[edge_index[0]], h[edge_index[1]]], dim=1)

        if self.mp_type in {
            "updownscale",
            "base",
            "updownscale_base",
        }:
            e = self.act(self.lin_geom(e))

        # --- Message Passing block --

        if self.mp_type == "updownscale" or self.mp_type == "updownscale_base":
            h = self.act(self.lin_down(h))  # downscale node rep.
            h = self.propagate(edge_index, x=h, W=e)  # propagate
            if self.graph_norm:
                h = self.act(self.graph_norm(h))
            h = self.act(self.lin_up(h))  # upscale node rep.

        elif self.mp_type == "updown_local_env":
            h = self.act(self.lin_down(h))
            chi = self.propagate(edge_index, x=h, W=e, local_env=True)
            e = self.lin_geom(e)
            h = self.propagate(edge_index, x=h, W=e)  # propagate
            if self.graph_norm:
                h = self.act(self.graph_norm(h))
            h = torch.cat((h, chi), dim=1)
            h = self.lin_up(h)

        elif self.mp_type in {"base", "simple"}:
            h = self.propagate(edge_index, x=h, W=e)  # propagate
            if self.graph_norm:
                h = self.act(self.graph_norm(h))
            h = self.act(self.lin_h(h))

        else:
            raise ValueError("mp_type provided does not exist")

        if self.complex_mp:
            h = self.act(self.other_mlp(h))

        return h

    def message(
        self,
        x_j: torch.Tensor,
        W: torch.Tensor,
        local_env=None,
    ) -> torch.Tensor:
        if local_env is not None:
            return W
        else:
            return x_j * W


class OutputBlock(nn.Module):
    """Compute task-specific predictions from final atom representations."""

    def __init__(
        self,
        energy_head: str,
        hidden_channels: int,
        act: callable,
        out_dim: int = 1,
    ) -> None:
        super().__init__()
        self.energy_head = energy_head
        self.act = act

        self.lin1 = Linear(hidden_channels, hidden_channels // 2)
        self.lin2 = Linear(hidden_channels // 2, out_dim)

        if self.energy_head == "weighted-av-final-embeds":
            self.w_lin = Linear(hidden_channels, 1)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin1.weight)
        self.lin1.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.lin2.weight)
        self.lin2.bias.data.fill_(0)
        if self.energy_head == "weighted-av-final-embeds":
            nn.init.xavier_uniform_(self.w_lin.weight)
            self.w_lin.bias.data.fill_(0)

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        batch,
        alpha: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass of the Output block.
        Called in FAENet to make prediction from final atom representations.

        Args:
            h (tensor): atom representations. (num_atoms, hidden_channels)
            edge_index (tensor): adjacency matrix. (2, num_edges)
            edge_weight (tensor): edge weights. (num_edges, )
            batch (tensor): batch indices. (num_atoms, )
            alpha (tensor): atom attention weights for late energy head. (num_atoms, )

        Returns:
            (tensor): graph-level representation (e.g. energy prediction)
        """
        if self.energy_head == "weighted-av-final-embeds":
            alpha = self.w_lin(h)

        # MLP
        h = self.lin1(h)
        h = self.act(h)
        h = self.lin2(h)

        if self.energy_head in {
            "weighted-av-initial-embeds",
            "weighted-av-final-embeds",
        }:
            h = h * alpha

        # Global pooling
        out = scatter(h, batch, dim=0, reduce="add")

        return out


class LambdaLayer(nn.Module):
    def __init__(self, func: callable):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


class ForceDecoder(nn.Module):
    """
    Predicts a force vector per atom from final atomic representations.

    Args:
        type (str): Type of force decoder to use
        input_channels (int): Number of input channels
        model_configs (dict): Dictionary of config parameters for the
            decoder's model
        act (callable): Activation function (NOT a module)

    Raises:
        ValueError: Unknown type of decoder

    Returns:
        (torch.Tensor): Predicted force vector per atom
    """

    def __init__(
        self,
        type: str,
        input_channels: int,
        model_configs: dict,
        act: callable,
    ):
        super().__init__()
        self.type = type
        self.act = act
        assert type in model_configs, f"Unknown type of force decoder: `{type}`"
        self.model_config = model_configs[type]
        if self.model_config.get("norm", "batch1d") == "batch1d":
            self.norm = lambda n: nn.BatchNorm1d(n)
        elif self.model_config["norm"] == "layer":
            self.norm = lambda n: nn.LayerNorm(n)
        elif self.model_config["norm"] in ["", None]:
            self.norm = lambda n: nn.Identity()
        else:
            raise ValueError(f"Unknown norm type: {self.model_config['norm']}")
        # Define the different force decoder models
        if self.type == "simple":
            assert "hidden_channels" in self.model_config
            self.model = nn.Sequential(
                nn.Linear(
                    input_channels,
                    self.model_config["hidden_channels"],
                ),
                LambdaLayer(act),
                nn.Linear(self.model_config["hidden_channels"], 3),
            )
        elif self.type == "mlp":  # from forcenet
            assert "hidden_channels" in self.model_config
            self.model = nn.Sequential(
                nn.Linear(
                    input_channels,
                    self.model_config["hidden_channels"],
                ),
                self.norm(self.model_config["hidden_channels"]),
                LambdaLayer(act),
                nn.Linear(self.model_config["hidden_channels"], 3),
            )
        elif self.type == "res":
            assert "hidden_channels" in self.model_config
            self.mlp_1 = nn.Sequential(
                nn.Linear(
                    input_channels,
                    input_channels,
                ),
                self.norm(input_channels),
                LambdaLayer(act),
            )
            self.mlp_2 = nn.Sequential(
                nn.Linear(
                    input_channels,
                    input_channels,
                ),
                self.norm(input_channels),
                LambdaLayer(act),
            )
            self.mlp_3 = nn.Sequential(
                nn.Linear(
                    input_channels,
                    self.model_config["hidden_channels"],
                ),
                self.norm(self.model_config["hidden_channels"]),
                LambdaLayer(act),
                nn.Linear(self.model_config["hidden_channels"], 3),
            )
        elif self.type == "res_updown":
            assert "hidden_channels" in self.model_config
            self.mlp_1 = nn.Sequential(
                nn.Linear(
                    input_channels,
                    self.model_config["hidden_channels"],
                ),
                self.norm(self.model_config["hidden_channels"]),
                LambdaLayer(act),
            )
            self.mlp_2 = nn.Sequential(
                nn.Linear(
                    self.model_config["hidden_channels"],
                    self.model_config["hidden_channels"],
                ),
                self.norm(self.model_config["hidden_channels"]),
                LambdaLayer(act),
            )
            self.mlp_3 = nn.Sequential(
                nn.Linear(
                    self.model_config["hidden_channels"],
                    input_channels,
                ),
                self.norm(input_channels),
                LambdaLayer(act),
            )
            self.mlp_4 = nn.Sequential(
                nn.Linear(
                    input_channels,
                    self.model_config["hidden_channels"],
                ),
                self.norm(self.model_config["hidden_channels"]),
                LambdaLayer(act),
                nn.Linear(self.model_config["hidden_channels"], 3),
            )
        else:
            raise ValueError(f"Unknown force decoder type: `{self.type}`")

        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.children():
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()
            else:
                if hasattr(layer, "weight"):
                    nn.init.xavier_uniform_(layer.weight)
                if hasattr(layer, "bias"):
                    layer.bias.data.fill_(0)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        if self.type == "res":
            return self.mlp_3(self.mlp_2(self.mlp_1(h)) + h)
        elif self.type == "res_updown":
            return self.mlp_4(self.mlp_3(self.mlp_2(self.mlp_1(h))) + h)
        return self.model(h)
