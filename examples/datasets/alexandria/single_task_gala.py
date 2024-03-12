from __future__ import annotations

import pytorch_lightning as pl
from torch.nn import LayerNorm, SiLU

from matsciml.lightning.data_utils import MatSciMLDataModule
from matsciml.models import GalaPotential
from matsciml.models.base import ScalarRegressionTask

model_args = {
    "D_in": 100,
    "hidden_dim": 128,
    "merge_fun": "concat",
    "join_fun": "concat",
    "invariant_mode": "full",
    "covariant_mode": "full",
    "include_normalized_products": True,
    "invar_value_normalization": "momentum",
    "eqvar_value_normalization": "momentum_layer",
    "value_normalization": "layer",
    "score_normalization": "layer",
    "block_normalization": "layer",
    "equivariant_attention": False,
    "tied_attention": True,
    "encoder_only": True,
}

mp_norms = {
    "formation_energy_per_atom_mean": -1.454,
    "formation_energy_per_atom_std": 1.206,
}

task = ScalarRegressionTask(
    mp_norms,
    encoder_class=GalaPotential,
    encoder_kwargs=model_args,
    output_kwargs={
        "norm": LayerNorm(128),
        "hidden_dim": 128,
        "activation": SiLU,
        "lazy": False,
        "input_dim": 128,
    },
    lr=1e-4,
    task_keys=["band_gap_ind"],
)


dm = MatSciMLDataModule.from_devset(
    "AlexandriaDataset",
    batch_size=5,
    num_workers=0,
)

trainer = pl.Trainer(fast_dev_run=10)
trainer.fit(task, datamodule=dm)
