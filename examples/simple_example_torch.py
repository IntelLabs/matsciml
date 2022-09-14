# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: MIT License

import pytorch_lightning as pl

import dgl, torch, os, sys
from torch import optim
from tqdm import tqdm

try:
    from ocpmodels.lightning.data_utils import S2EFDGLDataModule, IS2REDGLDataModule
    from ocpmodels.models import S2EFLitModule, IS2RELitModule, DimeNetPP
except:
    dir_path = os.path.dirname(os.path.realpath(__file__))
    sys.path.append("{}/../".format(dir_path))

    from ocpmodels.lightning.data_utils import S2EFDGLDataModule, IS2REDGLDataModule
    from ocpmodels.models import S2EFLitModule, IS2RELitModule, DimeNetPP


"""
simple_example_torch.py

This script was written to illustrate how to use the Open MatSci ML Toolkit
 pipeline in a way closely resembling the conventional/manual PyTorch way,
while reusing as much of the Lightning components as possible.

While we don't foresee many needing to do this, it does provide some
flexibility in how the pipeline could be modified, and break out
of the PyTorch Lightning mold if necessary to do some modular testing
and debugging (e.g. when you don't know if its your model or Lightning
causing issues).
"""

### Hardcoded settings for testing purposes
SEED = 42
BATCH_SIZE = 16
NUM_WORKERS = 0
MAX_STEPS = 300
REGRESS_FORCES = False

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
# create the S2EF task; lr and gamma are inconsequential because we create
# our own optimizer below
model = S2EFLitModule(dpp, regress_forces=REGRESS_FORCES, lr=1e-3, gamma=0.1)


# grab the devset; we will create our own data loader but we can rely
# on the `DataModule` to grab splits
data_module = S2EFDGLDataModule.from_devset()

data_module.setup()
# get the training split from the module
train_split = data_module.data_splits.get("train")
train_loader = train_split.data_loader(
    train_split,
    shuffle=True,
    num_workers=NUM_WORKERS,
    batch_size=BATCH_SIZE,
    collate_fn=data_module.collate_fn,  # data module knows how to batch
)

opt = optim.Adam(model.parameters(), lr=1e-3)

total_epochs = 2

print("Model Device: ", model.device)
print("S2EF Training")

# write the optimization loop manually
for epoch in range(total_epochs):
    # loop through the training set
    loop = tqdm(train_loader)
    for i, batch in enumerate(loop):
        if i < MAX_STEPS:
            opt.zero_grad()
            # reuse the logic implemented in compute losses
            loss = model._compute_losses(batch, i)
            # get the loss and backprop
            total_loss = loss.get("loss")
            total_loss.backward()
            opt.step()

            loop.set_description(f"Training Epoch {epoch}")
            loop.set_postfix(loss=total_loss.item())

        else:
            break


print("IS2RE Training")

model = IS2RELitModule(dpp, lr=1e-3, gamma=0.1)


# grab the devset; we will create our own data loader but we can rely
# on the `DataModule` to grab splits
data_module = IS2REDGLDataModule.from_devset()

data_module.setup()
# get the training split from the module
train_split = data_module.data_splits.get("train")
train_loader = train_split.data_loader(
    train_split,
    shuffle=True,
    num_workers=NUM_WORKERS,
    batch_size=BATCH_SIZE,
    collate_fn=data_module.collate_fn,  # data module knows how to batch
)

opt = optim.Adam(model.parameters(), lr=1e-3)

total_epochs = 2

# write the optimization loop manually
for epoch in range(total_epochs):
    # loop through the training set
    loop = tqdm(train_loader)
    for i, batch in enumerate(loop):
        if i < MAX_STEPS:
            opt.zero_grad()
            # reuse the logic implemented in compute losses
            loss = model._compute_losses(batch, i)
            # get the loss and backprop
            total_loss = loss.get("loss")
            total_loss.backward()
            opt.step()

            loop.set_description(f"Training Epoch {epoch}")
            loop.set_postfix(loss=total_loss.item())

        else:
            break
