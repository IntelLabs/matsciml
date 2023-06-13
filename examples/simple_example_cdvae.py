# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: MIT License
import sys, os
import pytorch_lightning as pl
from functools import partial
from pathlib import Path
import torch
from tqdm import tqdm

# from ocpmodels.lightning.data_utils import MaterialsProjectDataModule
# from ocpmodels.datasets.materials_project import (
#     DGLMaterialsProjectDataset,
# )
try:
    from ocpmodels.models.diffusion_pipeline import GenerationTask
    from ocpmodels.datasets.cdvae_datasets import CrystDataset, TensorCrystDataset
    from ocpmodels.datasets.cdvae_datamodule import CrystDataModule
    from ocpmodels.models.pyg.gemnet.decoder import GemNetTDecoder
    from ocpmodels.models.pyg.dimenetpp_wrap_cdvae import DimeNetPlusPlusWrap
    from examples.cdvae_configs import (
        enc_config, dec_config, cdvae_config, carbon_config,
        perov_config, mp20_config, mp_config
    )
    from ocpmodels.lightning.data_utils import MaterialsProjectDataModule
    from ocpmodels.datasets.materials_project import DGLMaterialsProjectDataset, PyGMaterialsProjectDataset, PyGCdvaeDataset
    from ocpmodels.models.diffusion_utils.data_utils import StandardScalerTorch

except:
    dir_path = os.path.dirname(os.path.realpath(__file__))
    sys.path.append("{}/../".format(dir_path))
    from ocpmodels.models.diffusion_pipeline import GenerationTask
    from ocpmodels.datasets.cdvae_datasets import CrystDataset, TensorCrystDataset
    from ocpmodels.datasets.cdvae_datamodule import CrystDataModule
    from ocpmodels.models.pyg.gemnet.decoder import GemNetTDecoder
    from ocpmodels.models.pyg.dimenetpp_wrap_cdvae import DimeNetPlusPlusWrap
    from ocpmodels.lightning.data_utils import MaterialsProjectDataModule
    from ocpmodels.datasets.materials_project import DGLMaterialsProjectDataset, PyGMaterialsProjectDataset, PyGCdvaeDataset
    from ocpmodels.models.diffusion_utils.data_utils import StandardScalerTorch

    from examples.cdvae_configs import (
        enc_config, dec_config, cdvae_config, carbon_config, 
        perov_config, mp20_config, mp_config
    )   

SCALER_LIMIT = 100

def get_scalers(datamodule):
    #loader = datamodule.train_dataloader()
    print("Building scalers")
    lattice_vals, prop_vals = [], []
    
    for i, data in tqdm(enumerate(datamodule.dataset)):
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
        if i > SCALER_LIMIT:
            break


    lattice_vals = torch.cat(lattice_vals)
    prop_vals = torch.cat(prop_vals)
    lattice_scaler = StandardScalerTorch()
    lattice_scaler.fit(lattice_vals)
    prop_scaler = StandardScalerTorch()
    prop_scaler.fit(prop_vals)

    return lattice_scaler, prop_scaler


# @hydra.main(version_base='1.1', config_path='../configs', config_name='regression_baseline')
def main():
    pl.seed_everything(1616)

    # include transforms to the data: shift center of mass and rescale magnitude of coordinates
    # dset = DGLMaterialsProjectDataset(
    #     "../materials_project/mp_data/base", transforms=[COMShift(), CoordinateScaling(0.1)]
    # )
    data_config = mp_config
    #data_config = carbon_config

    # init dataset-specific params in encoder/decoder
    enc_config['num_targets'] = cdvae_config['latent_dim'] #data_config['num_targets'] 
    enc_config['otf_graph'] = data_config['otf_graph']
    enc_config['readout'] = data_config['readout']

    cdvae_config['max_atoms'] = data_config['max_atoms']
    cdvae_config['teacher_forcing_max_epoch'] = data_config['teacher_forcing_max_epoch']
    cdvae_config['lattice_scale_method'] = data_config['lattice_scale_method']

    # dataset = PyGMaterialsProjectDataset()
    dm = MaterialsProjectDataModule(
        dataset=PyGCdvaeDataset(Path("/Users/mgalkin/git/projects.research.chem-ai.open-catalyst-collab/data/mp_data/train/")),
        val_split=0.1,
        # train_path=PyGMaterialsProjectDataset(Path("/Users/mgalkin/git/projects.research.chem-ai.open-catalyst-collab/data/mp_data/train/")),
        # val_split=PyGMaterialsProjectDataset(Path("/Users/mgalkin/git/projects.research.chem-ai.open-catalyst-collab/data/mp_data/val/")),
        # test_split=PyGMaterialsProjectDataset(Path("/Users/mgalkin/git/projects.research.chem-ai.open-catalyst-collab/data/mp_data/test/")),
        batch_size=4,
    )
    lattice_scaler, prop_scaler = get_scalers(dm)
    dm.dataset.lattice_scaler = lattice_scaler.copy()
    dm.dataset.scaler = prop_scaler.copy()


    # dataclass = partial(CrystDataset, **data_config)
    # splits = [
    #     dataclass(path=f"{data_config['root_path']}/{split}.csv") for split in ['train', 'val', 'test'] 
    # ]
    # dm = CrystDataModule(
    #     train=splits[0], 
    #     valid=splits[1],
    #     test=splits[2],
    #     num_workers=0,
    #     batch_size=4
    # )


    encoder = DimeNetPlusPlusWrap(**enc_config)
    decoder = GemNetTDecoder(**dec_config)
    model = GenerationTask(
        encoder=encoder, 
        decoder=decoder,
        **cdvae_config
    )
    # model.lattice_scaler = dm.lattice_scaler.copy()
    # model.scaler = dm.scaler.copy()

    #print(f"Passing scaler from datamodule to model <{dm.scaler}>")
    model.lattice_scaler = lattice_scaler.copy()
    model.scaler = prop_scaler.copy()


    # torch.save(dm.lattice_scaler, hydra_dir / 'lattice_scaler.pt')
    # torch.save(dm.scaler, hydra_dir / 'prop_scaler.pt')

    # # use the GNN in the LitModule for all the logging, loss computation, etc.
    # model = IS2RELitModule(dpp, lr=1e-3, gamma=0.1)
    # data_module = IS2REDGLDataModule.from_devset(
    #     batch_size=BATCH_SIZE, num_workers=NUM_WORKERS
    # )

    print('Training')
    trainer = pl.Trainer(accelerator="cpu", #strategy="ddp", 
                        devices=1, max_epochs=5)

    trainer.fit(model, datamodule=dm)

if __name__ == "__main__":
    main()