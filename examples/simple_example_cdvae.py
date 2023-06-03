# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: MIT License
import sys, os
import pytorch_lightning as pl
from functools import partial

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
        perov_config, mp20_config
    )
    from ocpmodels.lightning.data_utils import MaterialsProjectDataModule
    from ocpmodels.datasets.materials_project import DGLMaterialsProjectDataset, PyGMaterialsProjectDataset

except:
    dir_path = os.path.dirname(os.path.realpath(__file__))
    sys.path.append("{}/../".format(dir_path))
    from ocpmodels.models.diffusion_pipeline import GenerationTask
    from ocpmodels.datasets.cdvae_datasets import CrystDataset, TensorCrystDataset
    from ocpmodels.datasets.cdvae_datamodule import CrystDataModule
    from ocpmodels.models.pyg.gemnet.decoder import GemNetTDecoder
    from ocpmodels.models.pyg.dimenetpp_wrap_cdvae import DimeNetPlusPlusWrap
    from ocpmodels.lightning.data_utils import MaterialsProjectDataModule
    from ocpmodels.datasets.materials_project import DGLMaterialsProjectDataset, PyGMaterialsProjectDataset

    from examples.cdvae_configs import (
        enc_config, dec_config, cdvae_config, carbon_config, 
        perov_config, mp20_config
    )   

# try:
#     from ocpmodels.datasets import s2ef_devset, is2re_devset
#     from ocpmodels.models import DimeNetPP, S2EFLitModule, IS2RELitModule
#     from ocpmodels.lightning.data_utils import S2EFDGLDataModule, IS2REDGLDataModule

# except:
#     dir_path = os.path.dirname(os.path.realpath(__file__))
#     sys.path.append("{}/../".format(dir_path))

#     from ocpmodels.datasets import s2ef_devset, is2re_devset
#     from ocpmodels.models import DimeNetPP, S2EFLitModule, IS2RELitModule
#     from ocpmodels.lightning.data_utils import S2EFDGLDataModule, IS2REDGLDataModule

# @hydra.main(version_base='1.1', config_path='../configs', config_name='regression_baseline')
def main():
    pl.seed_everything(1616)

    # include transforms to the data: shift center of mass and rescale magnitude of coordinates
    # dset = DGLMaterialsProjectDataset(
    #     "../materials_project/mp_data/base", transforms=[COMShift(), CoordinateScaling(0.1)]
    # )
    data_config = carbon_config

    # init dataset-specific params in encoder/decoder
    enc_config['num_targets'] = cdvae_config['latent_dim'] #data_config['num_targets'] 
    enc_config['otf_graph'] = data_config['otf_graph']
    enc_config['readout'] = data_config['readout']

    cdvae_config['max_atoms'] = data_config['max_atoms']
    cdvae_config['teacher_forcing_max_epoch'] = data_config['teacher_forcing_max_epoch']
    cdvae_config['lattice_scale_method'] = data_config['lattice_scale_method']

    #dataset = PyGMaterialsProjectDataset()
    dm = MaterialsProjectDataModule.from_devset(pyg=True)
    dm.dataset.target_keys_list = ['formation_energy_per_atom']


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

    

    #dm = MaterialsProjectDataModule(dset, batch_size=32)

    encoder = DimeNetPlusPlusWrap(**enc_config)
    decoder = GemNetTDecoder(**dec_config)
    model = GenerationTask(
        encoder=encoder, 
        decoder=decoder,
        **cdvae_config
    )

    # print(f"Passing scaler from datamodule to model <{dm.scaler}>")
    # model.lattice_scaler = dm.lattice_scaler.copy()
    # model.scaler = dm.scaler.copy()


    # torch.save(dm.lattice_scaler, hydra_dir / 'lattice_scaler.pt')
    # torch.save(dm.scaler, hydra_dir / 'prop_scaler.pt')

    # # use the GNN in the LitModule for all the logging, loss computation, etc.
    # model = S2EFLitModule(dpp, regress_forces=REGRESS_FORCES, lr=1e-3, gamma=0.1)
    # data_module = S2EFDGLDataModule.from_devset(
    #     batch_size=BATCH_SIZE, num_workers=NUM_WORKERS
    # )

    # # alternatively, if you don't want to run with validation, just do S2EFDGLDataModule.from_devset
    # data_module = S2EFDGLDataModule(
    #     train_path=s2ef_devset,
    #     val_path=s2ef_devset,
    #     batch_size=BATCH_SIZE,
    #     num_workers=NUM_WORKERS,
    # )

    # # use the GNN in the LitModule for all the logging, loss computation, etc.
    # model = IS2RELitModule(dpp, lr=1e-3, gamma=0.1)
    # data_module = IS2REDGLDataModule.from_devset(
    #     batch_size=BATCH_SIZE, num_workers=NUM_WORKERS
    # )


    # data_module = IS2REDGLDataModule(
    #     train_path=is2re_devset,
    #     val_path=is2re_devset,
    #     batch_size=BATCH_SIZE,
    #     num_workers=NUM_WORKERS,
    # )

    print('Training')
    trainer = pl.Trainer(accelerator="cpu", #strategy="ddp", 
                        devices=1, max_epochs=5)

    trainer.fit(model, datamodule=dm)

if __name__ == "__main__":
    main()