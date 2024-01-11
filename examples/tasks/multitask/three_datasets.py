from __future__ import annotations

import pytorch_lightning as pl
from torch.nn import LayerNorm, SiLU

from matsciml.datasets import IS2REDataset, LiPSDataset, S2EFDataset
from matsciml.datasets.multi_dataset import MultiDataset
from matsciml.datasets.transforms import PointCloudToGraphTransform
from matsciml.lightning import callbacks as cb
from matsciml.lightning.data_utils import MultiDataModule
from matsciml.models import PLEGNNBackbone
from matsciml.models.base import (
    ForceRegressionTask,
    MultiTaskLitModule,
    ScalarRegressionTask,
)

pl.seed_everything(1616)

# LiPS data are saved as point clouds, and need to be converted to graphs
lips_dset = LiPSDataset.from_devset(
    transforms=[PointCloudToGraphTransform(backend="dgl", cutoff_dist=10.0)],
)
# OCP datasets are saved as DGL graphs
is2re_dset = IS2REDataset.from_devset()
s2ef_dset = S2EFDataset.from_devset()
# use MultiDataset to concatenate each dataset
dset = MultiDataset([lips_dset, is2re_dset, s2ef_dset])

# wrap multidataset in Lightning abstraction
dm = MultiDataModule(train_dataset=dset, batch_size=16)

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

model = PLEGNNBackbone(**model_args)
# shared output head arguments.
output_kwargs = {
    "norm": LayerNorm(128),
    "num_hidden": 2,
    "dropout": 0.2,
    "hidden_dim": 128,
    "activation": SiLU,
    "lazy": False,
    "input_dim": 128,
}
# add normalization to targets
is2re_norm = {
    "energy_relaxed_mean": -1.3314,
    "energy_relaxed_std": 2.2805,
    "energy_init_mean": 5.4111,
    "energy_init_std": 5.4003,
}
# these values will differ depending on how the data was preprocessed
s2ef_norm = {"energy_mean": -364.9521, "energy_std": 233.8758}
lips_norm = {"energy_mean": -357.6045, "energy_std": 0.5468}

# build three individual tasks; encoder will be shared later
r_is2re = ScalarRegressionTask(
    model,
    lr=1e-3,
    output_kwargs=output_kwargs,
    normalize_kwargs=is2re_norm,
    task_keys=["energy_relaxed"],
)
r_s2ef = ForceRegressionTask(
    model,
    lr=1e-3,
    output_kwargs=output_kwargs,
    normalize_kwargs=s2ef_norm,
    task_keys=["force"],
)
r_lips = ForceRegressionTask(
    model,
    lr=1e-3,
    output_kwargs=output_kwargs,
    normalize_kwargs=lips_norm,
    task_keys=["force"],
)

# initialize multitask with regression and classification on materials project and OCP
task = MultiTaskLitModule(
    ("IS2REDataset", r_is2re),
    ("S2EFDataset", r_s2ef),
    ("LiPSDataset", r_lips),
)

# using manual optimization for multitask, so "grad_clip" args do not work for trainer
trainer = pl.Trainer(
    overfit_batches=10,
    logger=False,
    enable_checkpointing=False,
    callbacks=[cb.GradientCheckCallback()],
)
trainer.fit(task, datamodule=dm)
