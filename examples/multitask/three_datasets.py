import pytorch_lightning as pl

from ocpmodels.datasets.materials_project import (
    DGLMaterialsProjectDataset,
)
from ocpmodels.datasets.multi_dataset import MultiDataset
from ocpmodels.datasets import IS2REDataset, is2re_devset, S2EFDataset, s2ef_devset
from ocpmodels.lightning.data_utils import MultiDataModule
from ocpmodels.models.base import (
    MultiTaskLitModule,
    ScalarRegressionTask,
    BinaryClassificationTask,
    ForceRegressionTask
)
from ocpmodels.models import PLEGNNBackbone
from ocpmodels.datasets.transforms import CoordinateScaling, COMShift

pl.seed_everything(1616)

# include transforms to the data: shift center of mass and rescale magnitude of coordinates
mp_dset = DGLMaterialsProjectDataset(
    "../materials_project/mp_data/base", transforms=[COMShift(), CoordinateScaling(0.1)]
)
is2re_dset = IS2REDataset(is2re_devset)
s2ef_dset = S2EFDataset(s2ef_devset)
# use MultiDataset to concatenate each dataset
dset = MultiDataset([mp_dset, is2re_dset, s2ef_dset])

# wrap multidataset in Lightning abstraction
dm = MultiDataModule(train_dataset=dset, batch_size=16)

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
# shared output head arguments.
output_kwargs = {
    "dropout": 0.2,
    "num_hidden": 2,
    "activation": "torch.nn.SiLU",
}
# normalization factors for Materials Project base dataset
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
# build tasks using joint encoder
r_mp = ScalarRegressionTask(
    model, lr=1e-3, output_kwargs=output_kwargs, normalize_kwargs=mp_norms
)
# no normalization for IS2RE and S2EF (energy only) but specified in the same way as above
r_is2re = ScalarRegressionTask(model, lr=1e-3, output_kwargs=output_kwargs)
r_s2ef = ForceRegressionTask(model, lr=1e-3, output_kwargs=output_kwargs)
c = BinaryClassificationTask(model, lr=1e-3, output_kwargs=output_kwargs)

# initialize multitask with regression and classification on materials project and OCP
task = MultiTaskLitModule(
    ("DGLMaterialsProjectDataset", r_mp),
    ("DGLMaterialsProjectDataset", c),
    ("IS2REDataset", r_is2re),
    ("S2EFDataset", r_s2ef),
)

# using manual optimization for multitask, so "grad_clip" args do not work for trainer
trainer = pl.Trainer(
    limit_train_batches=100,  # limit batches not max steps, since there are multiple optimizers
    logger=False,
    enable_checkpointing=False,
)
trainer.fit(task, datamodule=dm)
