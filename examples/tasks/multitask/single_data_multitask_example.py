from __future__ import annotations

import pytorch_lightning as pl
from torch.nn import L1Loss, LayerNorm, SiLU

from matsciml.datasets import MaterialsProjectDataset
from matsciml.datasets.transforms import (
    COMShift,
    CoordinateScaling,
    PointCloudToGraphTransform,
)
from matsciml.lightning import MatSciMLDataModule
from matsciml.models import PLEGNNBackbone
from matsciml.models.base import (
    BinaryClassificationTask,
    MultiTaskLitModule,
    ScalarRegressionTask,
)

pl.seed_everything(1616)

# configure a materials project data module, include transforms to
# DGL graphs, shift to center of mass, and rescale coordinate magnitudes
dm = MatSciMLDataModule(
    "MaterialsProjectDataset",
    train_path="./matsciml/datasets/materials_project/devset-full",
    dset_kwargs={
        "transforms": [
            PointCloudToGraphTransform("dgl", cutoff_dist=20.0),
            COMShift(),
            CoordinateScaling(0.1),
        ],
    },
    batch_size=32,
)

# configure EGNN
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

# shared output head arguments
output_kwargs = {
    "norm": LayerNorm(128),
    "num_hidden": 2,
    "dropout": 0.2,
    "hidden_dim": 128,
    "activation": SiLU,
    "lazy": False,
    "input_dim": 128,
}
# set target normalization valkues
mp_norms = {
    "band_gap_mean": 1.0761,
    "band_gap_std": 1.5284,
    "uncorrected_energy_per_atom_mean": -5.8859,
    "uncorrected_energy_per_atom_std": 1.7386,
    "formation_energy_per_atom_mean": -1.4762,
    "formation_energy_per_atom_std": 1.2009,
    "efermi_mean": 3.0530,
    "efermi_std": 2.7154,
    "energy_per_atom_mean": 6.2507,
    "energy_per_atom_std": 1.8614,
}
# configure regression and classification tasks, reading the
# target keys from the data module directly
r = ScalarRegressionTask(
    encoder_class=PLEGNNBackbone,
    encoder_kwargs=model_args,
    lr=1e-3,
    loss_func=L1Loss,
    output_kwargs=output_kwargs,
    normalize_kwargs=mp_norms,
    task_keys=["band_gap"],
)
c = BinaryClassificationTask(
    encoder_class=PLEGNNBackbone,
    encoder_kwargs=model_args,
    lr=1e-3,
    output_kwargs=output_kwargs,
    task_keys=["is_metal"],
)

# initialize multitask with regression and classification on materials project
task = MultiTaskLitModule(
    ("MaterialsProjectDataset", r),
    ("MaterialsProjectDataset", c),
)

# do steps without logging and checkpointing
trainer = pl.Trainer(fast_dev_run=100)
trainer.fit(task, datamodule=dm)
