from experiments.data_config import norm_dict

s2ef = {
    "dataset": "S2EFDataset",
    "debug": {
        "batch_size": 4,
        "num_workers": 0,
    },
    "experiment": {
        "train_path": "/datasets-alt/open-catalyst/s2ef_train_200K/ref_energy_s2ef_train_200K_dgl_munch_edges/",
        "val_split": "/datasets-alt/open-catalyst/s2ef_val_id/ref_energy_munch_s2ef_val_id/",
        "normalize_kwargs": norm_dict["s2ef"],
    },
    "target_keys": ["energy", "force"],
}
