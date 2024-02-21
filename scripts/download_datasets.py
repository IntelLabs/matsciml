from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from matsciml.datasets.carolina_db import CMDRequest
from matsciml.datasets.materials_project import MaterialsProjectRequest
from matsciml.datasets.nomad import NomadRequest
from matsciml.datasets.oqmd import OQMDRequest
from matsciml.datasets.alexandria import AlexandriaRequest


def main(args):
    base_data_dir = args.base_dataset_dir
    for dataset in args.dataset:
        if dataset == "Alexandria_3D_scan":
            indices = list(range(0, 5))
            request = AlexandriaRequest(
                indices, base_data_dir.joinpath("alexandria_3D_scan"), dataset="scan"
            )
            request.download_and_write(n_jobs=5)

        if dataset == "Alexandria_3D_pbesol":
            indices = list(range(0, 5))
            request = AlexandriaRequest(
                indices,
                base_data_dir.joinpath("alexandria_3D_pbesol"),
                dataset="pbesol",
            )
            request.download_and_write(n_jobs=5)

        if dataset == "Alexandria_3D_pbe":
            indices = list(range(0, 45))
            request = AlexandriaRequest(
                indices, base_data_dir.joinpath("alexandria_3D_pbe"), dataset="pbe"
            )
            request.download_and_write(n_jobs=5)

        if dataset == "Alexandria_2D":
            indices = list(range(0, 2))
            request = AlexandriaRequest(
                indices, base_data_dir.joinpath("alexandria_2D"), dataset="2D"
            )
            request.download_and_write(n_jobs=2)

        if dataset == "Alexandria_1D":
            indices = list(range(0, 1))
            request = AlexandriaRequest(
                indices, base_data_dir.joinpath("alexandria_1D"), dataset="1D"
            )
            request.download_and_write(n_jobs=1)

        if dataset == "CMD":
            cmd = CMDRequest(base_data_dir=base_data_dir.joinpath("carolina_db"))
            cmd.download_data()
            cmd.process_data()

        if dataset == "Nomad":
            nomad = NomadRequest(base_data_dir=base_data_dir.joinpath("nomad"))
            nomad.fetch_ids()
            nomad.split_files = [base_data_dir.joinpath("nomad", "all.yml")]
            nomad.download_data()

        if dataset == "OQMD":
            oqmd = OQMDRequest(base_data_dir=base_data_dir.joinpath("oqmd"))
            oqmd.download_data()
            oqmd.process_json()
            oqmd.to_lmdb(oqmd.data_dir)

        if dataset == "MaterialsProject":
            parameters = {
                "fields": [
                    "band_gap",
                    "structure",
                    "formula_pretty",
                    "efermi",
                    "symmetry",
                    "is_metal",
                    "is_magnetic",
                    "is_stable",
                    "formation_energy_per_atom",
                    "uncorrected_energy_per_atom",
                    "energy_per_atom",
                ],
            }
            api_kwargs = {
                "num_sites": (2, 10000),
                "num_chunks": 200,
                "chunk_size": 1000,
            }
            split_files = [
                "../matsciml/datasets/materials_project/train.yml",
                "../matsciml/datasets/materials_project/test.yml",
                "../matsciml/datasets/materials_project/val.yml",
            ]
            for file in split_files:
                with open(file) as f:
                    data = yaml.safe_load(f)
                    split, ids = next(iter(data.keys())), next(iter(data.values()))

                mp = MaterialsProjectRequest(
                    material_ids=ids,
                    **parameters,
                    **api_kwargs,
                )
                mp.retrieve_data()
                name = base_data_dir.joinpath(split)
                mp.to_lmdb(name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        nargs="*",
        help="String names of datasets to download (CMD, Nomad, OQMD, MaterialsProject).",
        required=True,
    )
    parser.add_argument(
        "--base_dataset_dir",
        type=Path,
        help="Root directory that holds Open MatSciML Toolkit data; typically same as OCP. (matsciml/datasets)",
        required=True,
    )
    args = parser.parse_args()
    main(args)
