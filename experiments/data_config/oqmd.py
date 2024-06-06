from experiments.data_config import norm_dict

oqmd = {
    "dataset": "OQMDDataset",
    "debug": {
        "batch_size": 4,
        "num_workers": 0,
    },
    "experiment": {
        "train_path": "/datasets-alt/molecular-data/oqmd/train",
        "val_split": "/datasets-alt/molecular-data/oqmd/val",
        "test_split": "/datasets-alt/molecular-data/oqmd/test",
        "normalize_kwargs": norm_dict["oqmd"],
    },
    "target_keys": ["band_gap", "energy", "stability"],
}
