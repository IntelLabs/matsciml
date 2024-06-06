from experiments.data_config import norm_dict

materials_project_trajectory = {
    "dataset": "MaterialsProjectDataset",
    "debug": {
        "batch_size": 16,
        "num_workers": 0,
    },
    "experiment": {
        "train_path": "/datasets-alt/molecular-data/mat_traj/mp-traj-full/train",
        "val_split": "/datasets-alt/molecular-data/mat_traj/mp-traj-full/val",
        "test_split": "/datasets-alt/molecular-data/mat_traj/mp-traj-full/test",
        "normalize_kwargs": norm_dict["mp-traj"],
        "batch_size": 16,
    },
    "target_keys": [
        "uncorrected_total_energy",
        "corrected_total_energy",
        "energy_per_atom",
        "ef_per_atom",
        "e_per_atom_relaxed",
        "ef_per_atom_relaxed",
        "force",
        "stress",
        "magmom",
        "bandgap",
    ],
}
