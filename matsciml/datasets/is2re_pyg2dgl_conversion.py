# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: MIT License
from __future__ import annotations

import os
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict, Union

import dgl
import lmdb
import torch
from torch_geometric.data import Data as PyGData
from tqdm import tqdm

from matsciml.datasets.generate_subsplit import connect_db_read, write_data


def convert_pyg_to_dgl(
    pyg_graph: PyGData,
) -> dict[str, dgl.DGLGraph | torch.Tensor]:
    # bijective mapping from PyG to DGL
    (u, v) = pyg_graph.edge_index
    dgl_graph = dgl.graph((u, v), num_nodes=pyg_graph.natoms)
    # loop over the node and edge attributes, and copy the data over
    for node_key in ["atomic_numbers", "force", "pos", "pos_relaxed", "fixed", "tags"]:
        dgl_graph.ndata[node_key] = getattr(pyg_graph, node_key)
    for edge_key in ["cell_offsets", "distances"]:
        dgl_graph.edata[edge_key] = getattr(pyg_graph, edge_key)
    return_data = {"graph": dgl_graph}
    for key in ["cell", "sid", "y_init", "y_relaxed", "natoms"]:
        return_data[key] = getattr(pyg_graph, key)
    return return_data


def main(args: Namespace):
    input_path = Path(args.pyg_lmdb)
    output_path = Path(args.output_folder)
    if not input_path.exists():
        raise FileNotFoundError(f"{input_path} could not be found.")
    if output_path.exists():
        raise ValueError(
            f"{output_path} already exists, please check its contents and remove the folder!",
        )
    os.makedirs(output_path)
    db_paths = sorted(input_path.glob("*.lmdb"))
    # loop over individual LMDB files within the input set, if there are
    # more than one
    for path in db_paths:
        target = output_path / path.name
        pyg_env = connect_db_read(path)
        # grab the keys
        with pyg_env.begin() as txn:
            keys = [key for key in txn.cursor().iternext(values=False)]
        # open the output file for writing
        target_env = lmdb.open(
            str(target),
            subdir=False,
            map_size=1099511627776 * 2,
            meminit=False,
            map_async=True,
        )
        with pyg_env.begin() as txn:
            for key in tqdm(keys):
                # for digit encodings, these are indexes of data points
                # that we want to convert
                if key.decode("utf-8").isdigit():
                    pyg_data = pickle.loads(txn.get(key))
                    dgl_data = convert_pyg_to_dgl(pyg_data)
                    # convert the key before writing
                    key = key.decode("utf-8")
                    write_data(key, dgl_data, target_env)
                # otherwise it's just metadata to copy over directly
                else:
                    metadata = pickle.loads(txn.get(key))
                    # convert the key before writing
                    key = key.decode("utf-8")
                    write_data(key, metadata, target_env)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--pyg_lmdb",
        "-i",
        type=str,
        help="Folder containing the PyG LMDB files to be converted.",
    )
    parser.add_argument(
        "--output_folder",
        "-o",
        type=str,
        help="Path to a non-existing folder to save DGL data to.",
    )
    args = parser.parse_args()
    main(args)
