from experiments.data_config import norm_dict

gnome = {
    "dataset": "GnomeMaterialsProjectDataset",
    "debug": {
        "batch_size": 4,
        "num_workers": 0,
    },
    "experiment": {
        "train_path": "/datasets-alt/molecular-data/gnome/train",
        "val_split": "/datasets-alt/molecular-data/gnome/val",
        "test_split": "/datasets-alt/molecular-data/data_lmdbs/gnome/test",
        "normalize_kwargs": norm_dict["gnome"],
    },
    "target_keys": ["corrected_total_energy", "force"],
}
