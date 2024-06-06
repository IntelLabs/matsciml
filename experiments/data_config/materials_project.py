from experiments.data_config import norm_dict

materials_project = {
    "dataset": "MaterialsProjectDataset",
    "debug": {
        "batch_size": 4,
        "num_workers": 0,
    },
    "experiment": {
        "train_path": "/datasets-alt/molecular-data/materials_project/train",
        "val_split": "/datasets-alt/molecular-data/materials_project/val",
        "test_split": "/datasets-alt/molecular-data/materials_project/test",
        "normalize_kwargs": norm_dict["materials-project"],
    },
    "target_keys": [
        "is_magnetic",
        "is_metal",
        "is_stable",
        "band_gap",
        "efermi",
        "energy_per_atom",
        "formation_energy_per_atom",
        "uncorrected_energy_per_atom",
        "symmetry_number",
        "symmetry_symbol",
        "symmetry_group",
    ],
}
