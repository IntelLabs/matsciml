from __future__ import annotations

import torch

mp_config = {
    "name": "Formation energy train",
    "root_path": "matsciml/datasets/mp_20",
    "prop": "formation_energy_per_atom",
    "num_targets": 1,
    "niggli": True,
    "primitive": False,
    "graph_method": "crystalnn",
    "lattice_scale_method": "scale_length",
    "preprocess_workers": 30,
    "readout": "mean",
    "max_atoms": 25,
    "otf_graph": False,
    "eval_model_name": "mp20",
    "train_max_epochs": 1000,
    "early_stopping_patience": 100000,
    "teacher_forcing_max_epoch": 500,
}

mp20_config = {
    "name": "Formation energy train",
    "root_path": "matsciml/datasets/mp_20",
    "prop": "formation_energy_per_atom",
    "num_targets": 1,
    "niggli": True,
    "primitive": False,
    "graph_method": "crystalnn",
    "lattice_scale_method": "scale_length",
    "preprocess_workers": 30,
    "readout": "mean",
    "max_atoms": 20,
    "otf_graph": False,
    "eval_model_name": "mp20",
    "train_max_epochs": 1000,
    "early_stopping_patience": 100000,
    "teacher_forcing_max_epoch": 500,
}

carbon_config = {
    "name": "energy per atom",
    "root_path": "matsciml/datasets/carbon_24",
    "prop": "energy_per_atom",
    "num_targets": 1,
    "niggli": True,
    "primitive": False,
    "graph_method": "crystalnn",
    "lattice_scale_method": "scale_length",
    "preprocess_workers": 30,
    "readout": "mean",
    "max_atoms": 24,
    "otf_graph": False,
    "eval_model_name": "carbon",
    "train_max_epochs": 4000,
    "early_stopping_patience": 100000,
    "teacher_forcing_max_epoch": 1000,
}

perov_config = {
    "name": "formation energy",
    "root_path": "matsciml/datasets/perov_5",
    "prop": "heat_ref",
    "num_targets": 1,
    "niggli": True,
    "primitive": False,
    "graph_method": "crystalnn",
    "lattice_scale_method": "scale_length",
    "preprocess_workers": 30,
    "readout": "mean",
    "max_atoms": 20,
    "otf_graph": False,
    "eval_model_name": "perovskite",
    "train_max_epochs": 3000,
    "early_stopping_patience": 100000,
    "teacher_forcing_max_epoch": 1500,
}

# Main CDVAE
cdvae_config = {
    "hidden_dim": 256,  # 256, 32 for debug
    "latent_dim": 256,  # 256, 32 for debug
    "fc_num_layers": 1,
    "max_atoms": None,  # dataset.max_atoms
    "cost_natom": 1.0,
    "cost_coord": 10.0,
    "cost_type": 1.0,
    "cost_lattice": 10.0,
    "cost_composition": 1.0,
    "cost_edge": 10.0,
    "cost_property": 1.0,
    "beta": 0.01,
    "teacher_forcing_lattice": True,
    "teacher_forcing_max_epoch": None,  # dataset.teacher_forcing_max_epoch
    "max_neighbors": 20,  # maximum number of neighbors for OTF graph bulding in decoder
    "radius": 12.0,  # maximum search radius for OTF graph building in decoder
    "sigma_begin": 10.0,
    "sigma_end": 0.01,
    "type_sigma_begin": 5.0,
    "type_sigma_end": 0.01,
    "num_noise_level": 50,
    "predict_property": False,
}

# default model configuration for DimeNetPPWrap
enc_config = {
    "hidden_channels": 128,  # 128, 32 for debug
    "out_emb_channels": 256,  # 256, 32 for debug
    "int_emb_size": 64,  # 64, 16 for debug
    "basis_emb_size": 8,
    "num_blocks": 4,
    "num_spherical": 7,
    "num_radial": 6,
    "cutoff": 7.0,
    "envelope_exponent": 5.0,
    # "activation": torch.nn.SiLU,
    "num_targets": None,  # data.num_targets
    "otf_graph": None,  # data.otf_graph
    "max_num_neighbors": 20,
    "num_before_skip": 1,
    "num_after_skip": 2,
    "num_output_layers": 3,
    "readout": None,  # data.readout
    # "regress_forces": False,
    # "num_atoms": None,
    # "bond_feat_dim": None,
}

# GemNetTDecoder
dec_config = {
    "hidden_dim": 128,  # default 128, 32 for debug
    "latent_dim": cdvae_config["latent_dim"],  # cdvae latent dim
    "max_neighbors": cdvae_config["max_neighbors"],  # model.max_neighbors
    "radius": cdvae_config["radius"],  # model.radius
    "scale_file": "matsciml/models/pyg/gemnet/gemnet-dT.json",  # json file
}
