# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: MIT License

import pytorch_lightning as pl

import torch, sys, os

try:
    from ocpmodels.datasets import s2ef_devset, is2re_devset
    from ocpmodels.models import DimeNetPP, S2EFLitModule, IS2RELitModule
    from ocpmodels.lightning.data_utils import S2EFDGLDataModule, IS2REDGLDataModule

except:
    dir_path = os.path.dirname(os.path.realpath(__file__))
    sys.path.append("{}/../".format(dir_path))

    from ocpmodels.datasets import s2ef_devset, is2re_devset
    from ocpmodels.models import DimeNetPP, S2EFLitModule, IS2RELitModule
    from ocpmodels.lightning.data_utils import S2EFDGLDataModule, IS2REDGLDataModule



BATCH_SIZE = 16
NUM_WORKERS = 4
REGRESS_FORCES = False

epochs = 5


# default model configuration for DimeNet++
model_config = {
    "emb_size": 128,
    "out_emb_size": 256,
    "int_emb_size": 64,
    "basis_emb_size": 8,
    "num_blocks": 2,
    "num_spherical": 7,
    "num_radial": 6,
    "cutoff": 10.0,
    "envelope_exponent": 5.0,
    "activation": torch.nn.SiLU,
}

# use default settings for DimeNet++
dpp = DimeNetPP(**model_config)

print('S2EF Training')

# use the GNN in the LitModule for all the logging, loss computation, etc.
model = S2EFLitModule(dpp, regress_forces=REGRESS_FORCES, lr=1e-3, gamma=0.1)
data_module = S2EFDGLDataModule.from_devset(
    batch_size=BATCH_SIZE, num_workers=NUM_WORKERS
)

# alternatively, if you don't want to run with validation, just do S2EFDGLDataModule.from_devset
data_module = S2EFDGLDataModule(
    train_path=s2ef_devset,
    val_path=s2ef_devset,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
)


trainer = pl.Trainer(accelerator="gpu", strategy="ddp", devices=2, max_epochs=epochs)

print('IS2RE Training')

trainer.fit(model, datamodule=data_module)



# use the GNN in the LitModule for all the logging, loss computation, etc.
model = IS2RELitModule(dpp, lr=1e-3, gamma=0.1)
data_module = IS2REDGLDataModule.from_devset(
    batch_size=BATCH_SIZE, num_workers=NUM_WORKERS
)


data_module = IS2REDGLDataModule(
    train_path=is2re_devset,
    val_path=is2re_devset,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
)

trainer = pl.Trainer(accelerator="gpu", strategy="ddp", devices=2, max_epochs=epochs)

trainer.fit(model, datamodule=data_module)