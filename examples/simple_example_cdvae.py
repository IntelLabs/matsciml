# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: MIT License
import sys, os
import pytorch_lightning as pl
from functools import partial
from pathlib import Path
import torch
from tqdm import tqdm

try:
    from ocpmodels.models.diffusion_pipeline import GenerationTask
    from ocpmodels.models.pyg.gemnet.decoder import GemNetTDecoder
    from ocpmodels.models.pyg.dimenetpp_wrap_cdvae import DimeNetPlusPlusWrap
    from examples.cdvae_configs import (
        enc_config, dec_config, cdvae_config, mp_config
    )
    from ocpmodels.lightning.data_utils import MaterialsProjectDataModule
    from ocpmodels.datasets.materials_project import DGLMaterialsProjectDataset, PyGMaterialsProjectDataset, PyGCdvaeDataset, CdvaeLMDBDataset
    from ocpmodels.models.diffusion_utils.data_utils import StandardScalerTorch

except:
    dir_path = os.path.dirname(os.path.realpath(__file__))
    sys.path.append("{}/../".format(dir_path))
    from ocpmodels.models.diffusion_pipeline import GenerationTask
    from ocpmodels.models.pyg.gemnet.decoder import GemNetTDecoder
    from ocpmodels.models.pyg.dimenetpp_wrap_cdvae import DimeNetPlusPlusWrap
    from ocpmodels.lightning.data_utils import MaterialsProjectDataModule
    from ocpmodels.datasets.materials_project import DGLMaterialsProjectDataset, PyGMaterialsProjectDataset, PyGCdvaeDataset, CdvaeLMDBDataset
    from ocpmodels.models.diffusion_utils.data_utils import StandardScalerTorch

    from examples.cdvae_configs import (
        enc_config, dec_config, cdvae_config, mp_config
    )   


def get_scalers(dataset):
    print("Building scalers")
    lattice_vals, prop_vals = [], []
    
    for i, data in tqdm(enumerate(dataset)):
        try:
            flag = len(data.keys()) == 1
            continue
        except:
            pass
        lengths = data['lengths']  # bs, 3
        num_atoms = data['num_nodes'] # bs
        angles = data['angles']
        lengths = lengths / float(num_atoms)**(1/3)
        ys = data['y']
        lattice_vals.append(torch.cat([lengths, angles], dim=-1))
        prop_vals.append(ys)

    lattice_vals = torch.cat(lattice_vals)
    prop_vals = torch.cat(prop_vals)
    lattice_scaler = StandardScalerTorch()
    lattice_scaler.fit(lattice_vals)
    prop_scaler = StandardScalerTorch()
    prop_scaler.fit(prop_vals)

    return lattice_scaler, prop_scaler


def main():
    pl.seed_everything(1616)
    data_config = mp_config

    # init dataset-specific params in encoder/decoder
    enc_config['num_targets'] = cdvae_config['latent_dim']
    enc_config['otf_graph'] = data_config['otf_graph']
    enc_config['readout'] = data_config['readout']

    cdvae_config['max_atoms'] = data_config['max_atoms']
    cdvae_config['teacher_forcing_max_epoch'] = data_config['teacher_forcing_max_epoch']
    cdvae_config['lattice_scale_method'] = data_config['lattice_scale_method']

    dm = MaterialsProjectDataModule(
        dataset=CdvaeLMDBDataset,
        train_path=Path("/Users/mgalkin/git/projects.research.chem-ai.open-catalyst-collab/data/cdvae_data/train/"),
        val_split=Path("/Users/mgalkin/git/projects.research.chem-ai.open-catalyst-collab/data/cdvae_data/val/"),
        test_split=Path("/Users/mgalkin/git/projects.research.chem-ai.open-catalyst-collab/data/cdvae_data/test/"),
        batch_size=256,
        num_workers=0,
    )
    # Load the data at the setup stage
    dm.setup()

    # Compute scalers for regression targets and lattice parameters
    lattice_scaler, prop_scaler = get_scalers(dm.splits['train'])
    dm.dataset.lattice_scaler = lattice_scaler.copy()
    dm.dataset.scaler = prop_scaler.copy()

    encoder = DimeNetPlusPlusWrap(**enc_config)
    decoder = GemNetTDecoder(**dec_config)
    model = GenerationTask(
        encoder=encoder, 
        decoder=decoder,
        **cdvae_config
    )

    model.lattice_scaler = lattice_scaler.copy()
    model.scaler = prop_scaler.copy()

    trainer = pl.Trainer(accelerator="cpu", #strategy="ddp", 
                        devices=1, max_epochs=1000, gradient_clip_val=1.0)

    trainer.fit(model, datamodule=dm)

if __name__ == "__main__":
    main()