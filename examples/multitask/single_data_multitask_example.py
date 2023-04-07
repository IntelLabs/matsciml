import pytorch_lightning as pl

from ocpmodels.lightning.data_utils import MaterialsProjectDataModule
from ocpmodels.datasets.materials_project import (
    DGLMaterialsProjectDataset,
)
from ocpmodels.models.base import (
    MultiTaskLitModule,
    ScalarRegressionTask,
    BinaryClassificationTask,
)
from ocpmodels.models import PLEGNNBackbone
from ocpmodels.datasets.transforms import CoordinateScaling, COMShift

pl.seed_everything(1616)

# include transforms to the data: shift center of mass and rescale magnitude of coordinates
dset = DGLMaterialsProjectDataset(
    "../materials_project/mp_data/base", transforms=[COMShift(), CoordinateScaling(0.1)]
)
dm = MaterialsProjectDataModule(dset, batch_size=32)

# configure EGNN
model_args = {
    "embed_in_dim": 1,
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
# shared output head arguments
output_kwargs = {
    "dropout": 0.2,
    "num_hidden": 2,
    "norm": "torch.nn.LazyBatchNorm1d",
    "activation": "torch.nn.SiLU",
}
r = ScalarRegressionTask(model, lr=1e-3, output_kwargs=output_kwargs)
c = BinaryClassificationTask(model, lr=1e-3, output_kwargs=output_kwargs)

# initialize multitask with regression and classification on materials project
task = MultiTaskLitModule(
    ("MaterialsProjectDataset", r),
    ("MaterialsProjectDataset", c),
)

# using manual optimization for multitask, so "grad_clip" args do not work for trainer
trainer = pl.Trainer(
    limit_train_batches=100,      # limit batches not max steps, since there are multiple optimizers
    logger=False,
    enable_checkpointing=False,
)
trainer.fit(task, datamodule=dm)
