"""
Original implementation by Matthew Spellings (Vector Institute) 5/25/2023

Modifications by Kelvin Lee to integrate into matsciml
"""
from __future__ import annotations

import argparse
import itertools
import os
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from joblib import Parallel, delayed
from tqdm import tqdm

from matsciml.datasets.symmetry.subgroup_classes import SubgroupGenerator
from matsciml.datasets.utils import connect_lmdb_write, write_lmdb_data

devset_kwargs = {
    "lmdb_path": "./devset",
    "batch_size": 1,
    "number": 1000,
    "symmetry": 12,
    "max_types": 100,
    "max_size": 40,
    "multilabel": False,
    "upsample": False,
    "normalize": True,
    "seed": 0,
    "lengthscale": 1.0,
    "filter_scale": 1e-2,
}
train_kwargs = {
    "lmdb_path": "./symmetry/train",
    "batch_size": 1,
    "number": 2000000,
    "symmetry": 12,
    "max_types": 100,
    "max_size": 40,
    "multilabel": False,
    "upsample": False,
    "normalize": True,
    "seed": 121,
    "lengthscale": 1.0,
    "filter_scale": 1e-2,
}
train_large_kwargs = deepcopy(train_kwargs)
train_large_kwargs["number"] = int(20_000_000)
train_large_kwargs["lmdb_path"] = "./symmetry/train_large"
val_kwargs = {
    "lmdb_path": "./symmetry/validation",
    "batch_size": 1,
    "number": 300000,
    "symmetry": 12,
    "max_types": 100,
    "max_size": 40,
    "multilabel": False,
    "upsample": False,
    "normalize": True,
    "seed": 2160,
    "lengthscale": 1.0,
    "filter_scale": 1e-2,
}


parser = argparse.ArgumentParser(description="Freeze a point group dataset")
parser.add_argument("lmdb_path", help="Filename to write", type=Path)
parser.add_argument(
    "-b",
    "--batch-size",
    type=int,
    default=1,
    help="Number of point clouds to place in a batch",
)
parser.add_argument(
    "-n",
    "--number",
    type=int,
    default=128,
    help="Number of point clouds to generate",
)
parser.add_argument(
    "-i",
    "--symmetry",
    type=int,
    default=12,
    help="Maximum axial symmetry degree to consider",
)
parser.add_argument(
    "-t",
    "--max-types",
    type=int,
    default=1,
    help="Maximum number of types to generate",
)
parser.add_argument(
    "-m",
    "--multilabel",
    type=bool,
    default=False,
    help="If True, use multilabel variant (group-subgroup relations)",
)
parser.add_argument(
    "-z",
    "--max-size",
    type=int,
    default=120,
    help="Maximum point cloud size",
)
parser.add_argument(
    "-u",
    "--upsample",
    type=bool,
    default=False,
    help="If True, upsample point clouds to max_size by randomly resampling points",
)
parser.add_argument("-s", "--seed", type=int, default=13, help="RNG seed to use")
parser.add_argument(
    "--normalize",
    type=bool,
    default=False,
    help="If True, normalize points to the surface of a sphere",
)
parser.add_argument(
    "-l",
    "--lengthscale",
    type=float,
    default=1.0,
    help="Lengthscale of generated point clouds",
)
parser.add_argument(
    "-f",
    "--filter-scale",
    type=float,
    default=1e-2,
    help="Distance to use for deduplicating symmetrized points",
)
parser.add_argument(
    "--devset",
    action="store_true",
    help="Override settings to generate the devset.",
)
parser.add_argument(
    "--train_set",
    action="store_true",
    help="Override settings to generate the train set.",
)
parser.add_argument(
    "--val_set",
    action="store_true",
    help="Override settings to generate the validation set.",
)
parser.add_argument(
    "--train_large_set",
    action="store_true",
    help="Override settings to generate the large (20M) train set.",
)
parser.add_argument(
    "--num_workers",
    type=int,
    default=1,
    help="Number of parallel workers to use, as well as number of LMDB files to save to.",
)


def generate_subgroup_data(index: int, lmdb_root: Path, **gen_kwargs) -> None:
    """
    Function for generating a set of point group data, intended
    to be used in parallel.

    Parameters
    ----------
    index : int
        Worker index, used to offset the LMDB file number
    lmdb_root : Path
        Root folder to dump LMDB data files to
    """
    config = deepcopy(gen_kwargs)
    seed = config["seed"] + index  # offset each worker by index
    del config["seed"]
    target_env = connect_lmdb_write(
        Path(lmdb_root).joinpath(f"data.{str(index).zfill(4)}.lmdb"),
    )
    # instantiate generator
    dataset = SubgroupGenerator(**config)
    generator = dataset.generate(seed)
    batches = itertools.islice(generator, 0, config["n_max"])
    for index, batch in tqdm(
        enumerate(batches),
        desc="Entries processed.",
        total=config["n_max"],
        position=index,
    ):
        # convert batch object into dict for pickling
        batch = batch._asdict()
        converted_dict = {}
        # loop over each point cloud property and convert them to tensors from NumPy
        for key, array in batch.items():
            # cast to single precision if it's double
            if isinstance(array, np.ndarray):
                if array.dtype == np.float64:
                    array = array.astype(np.float32)
                # for node types, cast to long
                if array.dtype == np.int32:
                    array = array.astype(np.int64)
                array = torch.from_numpy(array.squeeze())
            converted_dict[key] = array
        write_lmdb_data(index, converted_dict, target_env)


def main(
    lmdb_path,
    batch_size,
    number,
    symmetry,
    max_types,
    multilabel,
    max_size,
    upsample,
    seed,
    normalize,
    lengthscale,
    filter_scale,
    num_workers: int = 1,
    **kwargs,
):
    num_per_worker = number // num_workers
    assert num_per_worker > 0, f"Invalid number of samples per worker: {num_per_worker}"
    config = {
        "n_max": num_per_worker,
        "sym_max": symmetry,
        "type_max": max_types,
        "max_size": max_size,
        "batch_size": batch_size,
        "upsample": upsample,
        "encoding_filter": filter_scale,
        "multilabel": multilabel,
        "normalize": normalize,
        "lengthscale": lengthscale,
        "seed": seed,
    }
    dupes = [deepcopy(config) for _ in range(num_workers)]
    os.makedirs(lmdb_path, exist_ok=True)
    with Parallel(num_workers) as p_env:
        _ = p_env(
            delayed(generate_subgroup_data)(i, lmdb_path, **dupe)
            for i, dupe in enumerate(dupes)
        )


if __name__ == "__main__":
    args = parser.parse_args()
    kwargs = vars(args)
    if args.devset:
        kwargs.update(devset_kwargs)
    elif args.train_set:
        kwargs.update(train_kwargs)
    elif args.val_set:
        kwargs.update(val_kwargs)
    elif args.train_large_set:
        kwargs.update(train_large_kwargs)
    main(**kwargs)
