# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: MIT License

import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary
import dgl
import torch


try:
    from ocpmodels.datasets import is2re_devset
    from ocpmodels.lightning.data_utils import IS2REDGLDataModule
    from ocpmodels.models import DimeNetPP, IS2RELitModule
except:
    import sys, os

    dir_path = os.path.dirname(os.path.realpath(__file__))
    sys.path.append("{}/../".format(dir_path))

    from ocpmodels.datasets import is2re_devset
    from ocpmodels.lightning.data_utils import IS2REDGLDataModule
    from ocpmodels.models import DimeNetPP, IS2RELitModule


### Hardcoded settings for testing purposes
SEED = 42
BATCH_SIZE = 16
NUM_WORKERS = 4
# change this path to where your data resides (folder containing LMDB files)
DATA_PATH = is2re_devset

# this sets the random seeds for all (i.e. torch, random, numpy) except DGL
pl.seed_everything(SEED)
dgl.seed(SEED)

# default model configuration for DimeNet++
model_config = {
    "emb_size": 128,
    "out_emb_size": 256,
    "int_emb_size": 64,
    "basis_emb_size": 8,
    "num_blocks": 3,
    "num_spherical": 7,
    "num_radial": 6,
    "cutoff": 10.0,
    "envelope_exponent": 5.0,
    "activation": torch.nn.SiLU,
}

# use default settings for DimeNet++
dpp = DimeNetPP(**model_config)
model = IS2RELitModule(dpp, lr=1e-3, gamma=0.1)

data_module = IS2REDGLDataModule(
    DATA_PATH,
    batch_size=BATCH_SIZE,  # these are inconsequential and just to fill
    num_workers=NUM_WORKERS,  # the args. We will use our own instantiated loader below.
)

# default is TensorBoardLogger, but here we log to CSV for illustrative
# purposes; see link below for list of supported loggers:
# https://pytorch-lightning.readthedocs.io/en/1.6.3/extensions/logging.html
logger = CSVLogger("lightning_logs", name="DimeNetPP")

# callbacks are passed as a list into `Trainer`; see link below for API
# https://pytorch-lightning.readthedocs.io/en/1.6.3/extensions/callbacks.html
ckpt_callback = ModelCheckpoint("model_checkpoints", save_top_k=5, monitor="train_loss")

trainer = pl.Trainer(
    accelerator="cpu",
    logger=logger,
    callbacks=[ckpt_callback, ModelSummary(max_depth=2)],
    max_steps=10,
    max_epochs=5,  # run 5 epochs of 10 steps each
    log_every_n_steps=1,  # much more frequent than needed, but here it's to demo
)

# run the training procedure
trainer.fit(model, datamodule=data_module)
