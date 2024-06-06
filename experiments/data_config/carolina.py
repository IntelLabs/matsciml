from experiments.data_config import norm_dict

carolina = {
    "dataset": "CMDataset",
    "debug": {
        "batch_size": 4,
        "num_workers": 0,
    },
    "experiment": {
        "train_path": "/datasets-alt/molecular-data/carolina_matdb/train",
        "val_split": "/datasets-alt/molecular-data/carolina_matdb/val",
        "test_split": "/datasets-alt/molecular-data/carolina_matdb/test",
        "normalize_kwargs": norm_dict["carolina"],
    },
    "target_keys": ["energy", "symmetry_number", "symmetry_symbol"],
}
