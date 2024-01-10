# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: MIT License
"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
from __future__ import annotations

import bisect
import logging
import pickle
import random
from pathlib import Path
from typing import Type, Union

import dgl
import lmdb
import munch
import numpy as np
import scipy.sparse as sp
import torch
from dgl.convert import graph as dgl_graph
from dgl.data import DGLDataset
from dgl.data.dgl_dataset import DGLDataset
from dgl.nn.pytorch.factory import KNNGraph
from torch.utils.data import Dataset

# from torch_geometric.data import Batch
# from matsciml.common.utils import pyg2_data_transform


def munch_to_dgl(munch_obj: munch.Munch, g: dgl_graph):
    exclude_list = ["edge_index", "natoms", "y"]

    natoms = int(munch_obj["natoms"])
    graph_variables = munch.Munch()

    graph_variables.label = munch_obj.y

    for key, value in munch_obj.items():
        # print('key ', key)

        if key in exclude_list:
            continue
        else:
            try:
                if len(value) == natoms:
                    g.ndata[key] = value
                else:
                    graph_variables[key] = value
            except TypeError:
                graph_variables[key] = value

    return g, graph_variables


class TrajectoryLmdbDataset(Dataset):
    r"""Dataset class to load from LMDB files containing relaxation trajectories.
    Useful for Structure to Energy & Force (S2EF) and Initial State to
    Relaxed State (IS2RS) tasks.

    Args:
        config (dict): Dataset configuration
        transform (callable, optional): Data transform function.
            (default: :obj:`None`)
    """

    def __init__(self, config, transform=None):
        super().__init__()
        self.config = config

        srcdir = Path(self.config["src"])
        db_paths = sorted(srcdir.glob("*.lmdb"))
        assert len(db_paths) > 0, f"No LMDBs found in '{srcdir}'"

        self.metadata_path = srcdir / "metadata.npz"

        self._keys, self.envs = [], []
        for db_path in db_paths:
            self.envs.append(self.connect_db(db_path))
            length = pickle.loads(self.envs[-1].begin().get(b"length"))
            self._keys.append(list(range(length)))

        keylens = [len(k) for k in self._keys]
        self._keylen_cumulative = np.cumsum(keylens).tolist()
        self.transform = transform
        self.num_samples = sum(keylens)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Figure out which db this should be indexed from.
        db_idx = bisect.bisect(self._keylen_cumulative, idx)
        # Extract index of element within that db.
        el_idx = idx
        if db_idx != 0:
            el_idx = idx - self._keylen_cumulative[db_idx - 1]
        assert el_idx >= 0

        # Return features.
        datapoint_pickled = (
            self.envs[db_idx]
            .begin()
            .get(f"{self._keys[db_idx][el_idx]}".encode("ascii"))
        )
        data_object = pyg2_data_transform(pickle.loads(datapoint_pickled))
        if self.transform is not None:
            data_object = self.transform(data_object)

        data_object.id = f"{db_idx}_{el_idx}"

        return data_object

    def connect_db(self, lmdb_path=None):
        env = lmdb.open(
            str(lmdb_path),
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=1,
        )
        return env

    def close_db(self):
        for env in self.envs:
            env.close()


class TrajectoryLmdbDataset_DGL(DGLDataset):
    r"""Dataset class to load from LMDB files containing relaxation trajectories.
    Useful for Structure to Energy & Force (S2EF) and Initial State to
    Relaxed State (IS2RS) tasks.

    Args:
        config (dict): Dataset configuration
        transform (callable, optional): Data transform function.
            (default: :obj:`None`)

    DGL Loading Process:
    1. Load the LMDB Dictionary (leverage existing function)
    2. Construct a Radius Graph or KNN Graph
    3. Fill in the relevant data based on the loaded LMD Object


    """

    def __init__(self, root_path: str | type[Path], name: str, transform=None):
        super().__init__(name=name)

        srcdir = Path(root_path)
        db_paths = sorted(srcdir.glob("*.lmdb"))
        assert len(db_paths) > 0, f"No LMDBs found in '{srcdir}'"

        self.metadata_path = srcdir / "metadata.npz"

        self._keys, self.envs = [], []
        for db_path in db_paths:
            self.envs.append(self.connect_db(db_path))
            length = pickle.loads(self.envs[-1].begin().get(b"length"))
            self._keys.append(list(range(length)))

        keylens = [len(k) for k in self._keys]
        self._keylen_cumulative = np.cumsum(keylens).tolist()
        self.transform = transform
        self.num_samples = sum(keylens)

    def __len__(self):
        return self.num_samples

    def download(self):
        pass

    def process(self):
        pass

    def __getitem__(self, idx):
        # Figure out which db this should be indexed from.
        db_idx = bisect.bisect(self._keylen_cumulative, idx)
        # Extract index of element within that db.
        el_idx = idx
        if db_idx != 0:
            el_idx = idx - self._keylen_cumulative[db_idx - 1]
        assert el_idx >= 0

        # Return features.
        datapoint_pickled = (
            self.envs[db_idx]
            .begin()
            .get(f"{self._keys[db_idx][el_idx]}".encode("ascii"))
        )

        munch_data = pickle.loads(datapoint_pickled)

        if "edge_index" in munch_data.keys():
            u = munch_data["edge_index"][0]
            v = munch_data["edge_index"][1]
            g = dgl_graph((u, v), num_nodes=munch_data["natoms"])

        g, graph_data_dict = munch_to_dgl(munch_data, g)

        if self.transform is not None:
            print("Transform Functions Not Yet Implemented for DGL")
            # data_object = self.transform(data_object)

        graph_data_dict.id = f"{db_idx}_{el_idx}"

        return g, graph_data_dict

    def connect_db(self, lmdb_path=None):
        env = lmdb.open(
            str(lmdb_path),
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=1,
        )
        return env

    def close_db(self):
        for env in self.envs:
            env.close()


def data_list_collater(data_list, otf_graph=False):
    batch = Batch.from_data_list(data_list)

    if not otf_graph:
        try:
            n_neighbors = []
            for i, data in enumerate(data_list):
                n_index = data.edge_index[1, :]
                n_neighbors.append(n_index.shape[0])
            batch.neighbors = torch.tensor(n_neighbors)
        except NotImplementedError:
            logging.warning(
                "LMDB does not contain edge index information, set otf_graph=True",
            )

    return batch


def data_list_collater_dgl(data_list, otf_graph=False):
    graphs, labels = zip(*data_list)
    batch = dgl.batch(graphs)
    return batch, labels


def data_list_collater_gaanet(data_list, otf_graph=None, pc_size=6, sample_size=10):
    """
    1. Get the DGL Data Lists
    2. Find out the molecule indexes inside the given structure
    3. Extract all the neighbors in the crystal of the catalyst
    4. Rebatch the structure for the pointclouds
        a) Make sure that the targets are aligned with the indexes inside the inputs
        b
    """
    graphs, targets = map(list, zip(*data_list))

    batch_size = len(data_list)
    max_num_nodes = max([g.num_nodes() for g in graphs])

    in_feats = 1

    batch_feat_list = []
    batch_pos_list = []
    batch_force_list = []

    for g in graphs:
        # g = graphs[0]

        g_feat_list = []
        g_pos_list = []
        g_force_list = []
        shape_list = []

        knn_graph = KNNGraph(pc_size)
        g_knn = knn_graph(g.ndata["pos"])

        edge_0 = g_knn.edges()[0].detach()
        edge_1 = g_knn.edges()[1].detach()

        l1 = [g.ndata["tags"] == 2]  # molecule indexes
        l2 = [g.ndata["tags"] != 2]  # substrate indexes
        mol_idx = g.nodes()[l1]
        substrate_idx = g.nodes()[l2]

        substrate_idx_sample_idx = random.sample(
            range(len(substrate_idx)),
            min(sample_size, len(substrate_idx)),
        )

        substrate_idx_sample = substrate_idx[substrate_idx_sample_idx]

        for subs_id in substrate_idx_sample:
            l3 = [edge_0 == subs_id]  # atom index
            l4 = [edge_1[l3]]  # neighbor indexes

            pc_feat = torch.cat(
                [
                    g.ndata["atomic_numbers"][subs_id].unsqueeze(-1),
                    g.ndata["atomic_numbers"][l4],
                    g.ndata["atomic_numbers"][mol_idx],
                ],
                dim=0,
            ).unsqueeze(-1)
            pc_pos = torch.cat(
                [
                    g.ndata["pos"][subs_id].unsqueeze(0),
                    g.ndata["pos"][l4],
                    g.ndata["pos"][mol_idx],
                ],
                dim=0,
            )
            pc_force = torch.cat(
                [
                    g.ndata["force"][subs_id].unsqueeze(0),
                    g.ndata["force"][l4],
                    g.ndata["force"][mol_idx],
                ],
                dim=0,
            )

            shape_list.append(pc_feat.shape[0])
            g_feat_list.append(pc_feat)
            g_pos_list.append(pc_pos)
            g_force_list.append(pc_force)

        # Need to align the sizes of the tensors (padding needed to create the batches)

        max_feat_size = max(shape_list)

        node_feats = torch.zeros(len(shape_list), max_feat_size, in_feats)
        positions = torch.zeros(len(shape_list), max_feat_size, 3)
        true_forces = torch.zeros(len(shape_list), max_feat_size, 3)

        for ii, (feats, pos, forces) in enumerate(
            zip(g_feat_list, g_pos_list, g_force_list),
        ):
            node_feats[ii][0 : shape_list[ii]] = feats
            positions[ii][0 : shape_list[ii]] = pos
            true_forces[ii][0 : shape_list[ii]] = forces

    batch_feat_tens = node_feats
    batch_pos_tens = positions
    batch_force_tens = true_forces

    return batch_feat_tens, batch_pos_tens, batch_force_tens, targets


def data_list_collater_gaanet_depr(data_list, otf_graph=None):
    graphs, targets = map(list, zip(*data_list))

    batch_size = len(data_list)
    max_num_nodes = max([g.num_nodes() for g in graphs])

    in_feats = 1

    node_feats = torch.zeros(batch_size, max_num_nodes, in_feats)
    positions = torch.zeros(batch_size, max_num_nodes, 3)
    true_forces = torch.zeros(batch_size, max_num_nodes, 3)

    for i in range(batch_size):
        num_nodes = graphs[i].num_nodes()

        node_feats[i, :num_nodes, :] = graphs[i].ndata["atomic_numbers"].unsqueeze(-1)
        positions[i, :num_nodes, :] = graphs[i].ndata["pos"]
        true_forces[i, :num_nodes, :] = graphs[i].ndata["force"]
    return node_feats, positions, true_forces, targets
