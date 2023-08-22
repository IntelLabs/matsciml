import argparse
from pathlib import Path

from ocpmodels.datasets.carolina_db import CMDRequest
from ocpmodels.datasets.nomad import NomadRequest
from ocpmodels.datasets.oqmd import OQMDRequest

def main(args):
    base_data_dir = args.base_dataset_dir
    for dataset in args.dataset:
        if dataset == "CMD":
            cmd = CMDRequest(base_data_dir=base_data_dir.joinpath("carolina_db"))
            cmd.download_data()
            cmd.process_data()

        if dataset == "Nomad":
            nomad = NomadRequest(base_data_dir=base_data_dir.joinpath("nomad"))
            nomad.fetch_ids()
            nomad.split_files = [base_data_dir.joinpath("nomad", "all")]
            nomad.download_data()

        if dataset == "OQMD":
            oqmd = OQMDRequest(base_data_dir=base_data_dir.joinpath("oqmd"))
            oqmd.download_data()
            oqmd.process_json()
            oqmd.to_lmdb(oqmd.data_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        nargs="*",
        help="String names of datasets to download (CMD, Nomad, OQMD).",
        required=True
    )
    parser.add_argument(
        "--base_dataset_dir",
        type=Path,
        help="Root directory that holds Open MatSciML Toolkit data; typically same as OCP. (ocpmodels/datasets)",
        required=True
    )
    args = parser.parse_args()
    main(args)
