from __future__ import annotations

import lightning.pytorch as pl
from torch.nn import LayerNorm, SiLU

from matsciml.datasets.transforms import (
    PointCloudToGraphTransform,
    PeriodicPropertiesTransform,
)
from matsciml.lightning.data_utils import MatSciMLDataModule
from matsciml.models import PLEGNNBackbone
from matsciml.models.base import ScalarRegressionTask

pl.seed_everything(21616)

model_args = {
    "embed_in_dim": 128,
    "embed_hidden_dim": 32,
    "embed_out_dim": 128,
    "embed_depth": 5,
    "embed_feat_dims": [128, 128, 128],
    "embed_message_dims": [128, 128, 128],
    "embed_position_dims": [64, 64],
    "embed_edge_attributes_dim": 0,
    "embed_activation": "relu",
    "embed_residual": True,
    "embed_normalize": True,
    "embed_tanh": True,
    "embed_activate_last": False,
    "embed_k_linears": 1,
    "embed_use_attention": False,
    "embed_attention_norm": "sigmoid",
    "readout": "sum",
    "node_projection_depth": 3,
    "node_projection_hidden_dim": 128,
    "node_projection_activation": "relu",
    "prediction_out_dim": 1,
    "prediction_depth": 3,
    "prediction_hidden_dim": 128,
    "prediction_activation": "relu",
    "encoder_only": True,
}

model = PLEGNNBackbone(**model_args)
task = ScalarRegressionTask(
    model,
    output_kwargs={
        "norm": LayerNorm(128),
        "hidden_dim": 128,
        "activation": SiLU,
        "lazy": False,
        "input_dim": 128,
    },
    lr=1e-3,
    task_keys=["band_gap_ind"],
)

dm = MatSciMLDataModule.from_devset(
    "AlexandriaDataset",
    dset_kwargs={
        "transforms": [
            PeriodicPropertiesTransform(10.0),
            PointCloudToGraphTransform(
                "dgl",
                cutoff_dist=10.0,
                node_keys=["pos", "atomic_numbers"],
            ),
        ],
    },
    batch_size=16,
    num_workers=0,
)

trainer = pl.Trainer(
    fast_dev_run=10,
)
trainer.fit(task, datamodule=dm)
