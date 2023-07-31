"""
Original implementation by Matthew Spellings (Vector Institute) 5/25/2023

Modifications by Kelvin Lee to integrate into matsciml
"""

import argparse
import itertools
import os
from pathlib import Path
from copy import deepcopy

import lmdb
import torch
import numpy as np
from tqdm import tqdm

from ocpmodels.datasets.symmetry.subgroup_classes import SubgroupGenerator
from ocpmodels.datasets.utils import write_lmdb_data, connect_lmdb_write


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
    "-n", "--number", type=int, default=128, help="Number of point clouds to generate"
)
parser.add_argument(
    "-i",
    "--symmetry",
    type=int,
    default=12,
    help="Maximum axial symmetry degree to consider",
)
parser.add_argument(
    "-t", "--max-types", type=int, default=1, help="Maximum number of types to generate"
)
parser.add_argument(
    "-m",
    "--multilabel",
    type=bool,
    default=False,
    help="If True, use multilabel variant (group-subgroup relations)",
)
parser.add_argument(
    "-z", "--max-size", type=int, default=120, help="Maximum point cloud size"
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
    "--devset", action="store_true", help="Override settings to generate the devset."
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
    config["seed"] += index  # offset each worker by index
    target_env = connect_lmdb_write(lmdb_root.joinpath(f"data.{index.zfill(4)}.lmdb"))
    # instantiate generator
    dataset = SubgroupGenerator(**config)
    generator = dataset.generate(config["seed"])
    batches = itertools.islice(generator, 0, config["number"])
    for index, batch in tqdm(
        enumerate(batches),
        desc="Entries processed.",
        total=config["number"],
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
):
    dataset = SubgroupGenerator(
        number,
        symmetry,
        max_types,
        max_size,
        batch_size,
        upsample,
        filter_scale,
        multilabel=multilabel,
        normalize=normalize,
        lengthscale=lengthscale,
    )

    # now prepare to dump the data
    if isinstance(lmdb_path, str):
        lmdb_path = Path(lmdb_path)

    os.makedirs(lmdb_path, exist_ok=True)
    target_env = lmdb.open(
        str(lmdb_path.joinpath("data.lmdb")),
        subdir=False,
        map_size=1099511627776 * 2,
        meminit=False,
        map_async=True,
    )
    generator = dataset.generate(seed)

    # batches = list(itertools.islice(generator, 0, number))
    batches = itertools.islice(generator, 0, number)
    for index, batch in tqdm(
        enumerate(batches), desc="Entries processed.", total=number
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


if __name__ == "__main__":
    args = parser.parse_args()
    if args.devset:
        config = devset_kwargs
    elif args.train_set:
        config = train_kwargs
    elif args.val_set:
        config = val_kwargs
    else:
        config = vars(args)
    main(**config)
