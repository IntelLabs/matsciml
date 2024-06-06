from experiments.data_config import norm_dict

is2re = {
    "dataset": "IS2REDataset",
    "debug": {
        "batch_size": 4,
        "num_workers": 0,
    },
    "experiment": {
        "train_path": "/datasets-alt/open-catalyst/carmelo_copy_is2re/is2re/all/train",
        "val_split": "/datasets-alt/open-catalyst/carmelo_copy_is2re/is2re/all/val_id",
        "normalize_kwargs": norm_dict["is2re"],
    },
    "target_keys": [
        "energy_init",
        "energy_relaxed",
    ],
}
