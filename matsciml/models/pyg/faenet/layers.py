from __future__ import annotations

import pandas as pd
import torch
import torch.nn as nn
from mendeleev.fetch import fetch_ionization_energies, fetch_table


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

    def __init__(self, props=True, props_grad=False, pg=False, short=False) -> None:
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
            self.group_size = df.group_id.unique().shape[0]
            group = torch.cat(
                [
                    torch.ones(1, dtype=torch.long),
                    torch.tensor(df.group_id.loc[:100].values, dtype=torch.long),
                ],
            )

            self.period_size = df.period.loc[:100].unique().shape[0]
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
            df = df.loc[:85, :]

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


import torch.nn as nn


class LambdaLayer(nn.Module):
    def __init__(self, func):
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

    def __init__(self, type, input_channels, model_configs, act):
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

    def forward(self, h):
        if self.type == "res":
            return self.mlp_3(self.mlp_2(self.mlp_1(h)) + h)
        elif self.type == "res_updown":
            return self.mlp_4(self.mlp_3(self.mlp_2(self.mlp_1(h))) + h)
        return self.model(h)
