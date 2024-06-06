from experiments.data_config import norm_dict

nomad = {
    "dataset": "NomadDataset",
    "debug": {
        "batch_size": 4,
        "num_workers": 0,
    },
    "experiment": {
        "train_path": "/datasets-alt/molecular-data/nomad/train",
        "val_split": "/datasets-alt/molecular-data/nomad/val",
        "test_split": "/datasets-alt/molecular-data/nomad/test",
        "normalize_kwargs": norm_dict["nomad"],
    },
    "target_keys": [
        "spin_polarized",
        "efermi",
        "relative_energy",
        "symmetry_number",
        "symmetry_symbol",
        "symmetry_group",
    ],
}
