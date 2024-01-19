# Copyright (C) 2022-3 Intel Corporation
# SPDX-License-Identifier: MIT License
from __future__ import annotations

import os
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import List

import lmdb
import numpy as np
from tqdm import tqdm

from matsciml.datasets import utils

"""
Script for generating subsplits from the original OCP datasets.
See the bottom of this script for usage and documentation.
"""


def generate_split_indices(
    all_indices: np.ndarray,
    splits_lengths: list[int],
) -> list[np.ndarray]:
    cum_splits = np.cumsum(splits_lengths)
    # check that the chunks we ask for do not exceed the length
    # of the actual array
    assert cum_splits.max() <= len(all_indices)
    # make the split
    splits = np.split(all_indices, cum_splits)
    # only return what we asked for
    return splits[: len(splits_lengths)]


def main(args: Namespace):
    seed = args.seed
    # make the random splitting deterministic
    rng = np.random.default_rng(seed)
    # connect to the data folder and find the *.lmdb files
    input_path = Path(args.lmdb_folder)
    if not input_path.exists():
        raise FileNotFoundError(f"{input_path} could not be found.")
    db_paths = sorted(input_path.glob("*.lmdb"))
    # now check that split names match the number of lengths specified
    assert len(args.lengths) == len(args.names)
    output_folders = [Path(name) for name in args.names]
    # make the output folders if they don't exist already
    for folder in output_folders:
        os.makedirs(folder, exist_ok=True)
    indices = []
    for index, path in enumerate(db_paths):
        length = utils.get_lmdb_data_length(path)
        # generate array of indices for lookup later
        indices.extend([(index, value) for value in range(length)])
    indices = np.array(indices)
    rng.shuffle(indices)
    # get our splits
    split_indices = generate_split_indices(indices, args.lengths)
    # initialize the lmdb environments for reading
    origin_envs = [utils.connect_db_read(path) for path in db_paths]
    # TODO open target LMDB files ready for writing
    for path, split in zip(output_folders, split_indices):
        output_env = lmdb.open(
            str(path.joinpath("data.0000.lmdb")),
            subdir=False,
            map_size=1099511627776 * 2,
            meminit=False,
            map_async=True,
        )
        # write out some metadata; how many graphs, and which split/file
        # and index it came from
        utils.write_data("length", len(split), output_env)
        utils.write_data("origin_file", input_path, output_env)
        utils.write_data("origin_indices", split, output_env)
        # copy each item into the new LMDB file
        for target_index, origin_index in enumerate(tqdm(split)):
            data = utils.get_data_from_index(
                origin_index[0],
                origin_index[1],
                origin_envs,
            )
            utils.write_data(target_index, data, output_env)


if __name__ == "__main__":
    doc = """
    Utility script to generate sub splits from one of the original OCP
    LMDB datasets. Point this to a folder containing the many LMDB files
    (e.g. data/s2ef/200k/train), specify the names of the different
    splits, and then the lengths of each split.

    Internally, we will randomly shuffle all of the indices of the
    original split, and divide them up into each split. This way, there
    is no leakage between each of the splits, and ostensibly can be
    used for quick cross-validation.

    A single LMDB file is created within the split name you specify,
    as the intention is to not actually have extremely large splits.
    Each key within the LMDB file corresponds to the graph, in addition
    to three other metadata keys; `length` (number of data points),
    `origin_file` (which split this subsplit is from), and `origin_indices`
    (the index mapping back to which file and index in the original split).
    """
    parser = ArgumentParser(description=doc)
    parser.add_argument(
        "--lmdb_folder",
        "-i",
        type=str,
        help="Folder containing LMDB files you wish to split.",
    )
    parser.add_argument(
        "--seed",
        "-s",
        type=int,
        default=150916,
        help="Random seed for reproducible splits.",
    )
    parser.add_argument(
        "--names",
        "-n",
        type=str,
        nargs="+",
        help="Space delimited names to assign to each split.",
    )
    parser.add_argument(
        "--lengths",
        "-l",
        type=int,
        nargs="+",
        help="Space delimited integers for the length of each split.",
    )
    args = parser.parse_args()
    main(args)
