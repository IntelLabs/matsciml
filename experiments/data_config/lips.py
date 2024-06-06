lips = {
    "dataset": "LiPSDataset",
    "debug": {
        "batch_size": 4,
        "num_workers": 0,
    },
    "experiment": {
        "train_path": "/datasets-alt/molecular-data/lips/train",
        "val_split": "/datasets-alt/molecular-data/lips/val",
        "test_split": "/datasets-alt/molecular-data/lips/test",
        "normalize_kwargs": norm_dict["lips"],
    },
    "target_keys": ["energy", "force"],
}
