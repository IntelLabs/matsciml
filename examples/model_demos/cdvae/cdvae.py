# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: MIT License
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pytorch_lightning as pl
import torch
from tqdm import tqdm

try:
    from examples.model_demos.cdvae.cdvae_configs import (
        cdvae_config,
        dec_config,
        enc_config,
        mp_config,
    )
    from matsciml.datasets.materials_project import CdvaeLMDBDataset
    from matsciml.lightning.data_utils import MatSciMLDataModule
    from matsciml.models.diffusion_pipeline import GenerationTask
    from matsciml.models.diffusion_utils.data_utils import StandardScalerTorch
    from matsciml.models.pyg.dimenetpp_wrap_cdvae import DimeNetPlusPlusWrap
    from matsciml.models.pyg.gemnet.decoder import GemNetTDecoder

except:
    dir_path = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(f"{dir_path}/../")
    from examples.cdvae_configs import cdvae_config, dec_config, enc_config, mp_config
    from examples.model_demos.cdvae.cdvae_configs import (
        cdvae_config,
        dec_config,
        enc_config,
        mp_config,
    )
    from matsciml.datasets.materials_project import CdvaeLMDBDataset
    from matsciml.lightning.data_utils import MatSciMLDataModule
    from matsciml.models.diffusion_pipeline import GenerationTask
    from matsciml.models.diffusion_utils.data_utils import StandardScalerTorch
    from matsciml.models.pyg.dimenetpp_wrap_cdvae import DimeNetPlusPlusWrap
    from matsciml.models.pyg.gemnet.decoder import GemNetTDecoder


# computing scalers to re-scale regression targets to a normalized range
def get_scalers(dataset):
    print("Building scalers")
    lattice_vals, prop_vals = [], []

    for i, data in tqdm(enumerate(dataset)):
        try:
            flag = len(data.keys()) == 1
            continue
        except:
            pass
        lengths = data["lengths"]  # bs, 3
        num_atoms = data["num_nodes"]  # bs
        angles = data["angles"]
        lengths = lengths / float(num_atoms) ** (1 / 3)
        ys = data["y"]
        lattice_vals.append(torch.cat([lengths, angles], dim=-1))
        prop_vals.append(ys)

    lattice_vals = torch.cat(lattice_vals)
    prop_vals = torch.cat(prop_vals)
    lattice_scaler = StandardScalerTorch()
    lattice_scaler.fit(lattice_vals)
    prop_scaler = StandardScalerTorch()
    prop_scaler.fit(prop_vals)

    return lattice_scaler, prop_scaler


"""
The main script to train CDVAE and its components (encoder, decoder, denoising).
Sampling and evaluation are performed in separate scripts (_inference, _metrics)
based on a saved trained checkpoint.
"""


def main(args):
    pl.seed_everything(1616)
    data_config = mp_config

    # init dataset-specific params in encoder/decoder
    enc_config["num_targets"] = cdvae_config["latent_dim"]
    enc_config["otf_graph"] = data_config["otf_graph"]
    enc_config["readout"] = data_config["readout"]

    cdvae_config["max_atoms"] = data_config["max_atoms"]
    cdvae_config["teacher_forcing_max_epoch"] = data_config["teacher_forcing_max_epoch"]
    cdvae_config["lattice_scale_method"] = data_config["lattice_scale_method"]

    # data preparation
    data_path = Path(args.data_path)
    dm = MatSciMLDataModule(
        dataset=CdvaeLMDBDataset,
        train_path=data_path / "train",
        val_split=data_path / "val",
        test_split=data_path / "test",
        batch_size=args.batch_size,
        num_workers=0,
    )
    # Load the data at the setup stage
    dm.setup()

    # Compute scalers for regression targets and lattice parameters
    # By default, we will train CDVAE on the formation energy
    lattice_scaler, prop_scaler = get_scalers(dm.splits["train"])
    dm.dataset.lattice_scaler = lattice_scaler.copy()
    dm.dataset.scaler = prop_scaler.copy()

    # create the encoder, decoder, and CDVAE generative model (GenerativeTask)
    encoder = DimeNetPlusPlusWrap(**enc_config)
    decoder = GemNetTDecoder(**dec_config)
    model = GenerationTask(encoder=encoder, decoder=decoder, **cdvae_config)

    model.lattice_scaler = lattice_scaler.copy()
    model.scaler = prop_scaler.copy()

    # create the trainer
    trainer = pl.Trainer(
        accelerator="cpu" if not args.gpu else "gpu",  # strategy="ddp",
        devices=1,
        max_epochs=args.epochs,
        gradient_clip_val=1.0,
    )

    # train CDVAE
    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)  # path to the LMDB sources
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--epochs", default=1000, type=int)
    parser.add_argument("--gpu", default=False, type=bool)
    args = parser.parse_args()
    main(args)
