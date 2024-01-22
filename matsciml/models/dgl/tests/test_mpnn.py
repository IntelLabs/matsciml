from __future__ import annotations

import dgl
import pytest
import torch

from matsciml.datasets import transforms
from matsciml.datasets.materials_project import MaterialsProjectDataset
from matsciml.models import MPNN

dgl.seed(215106)


@pytest.fixture
def test_batch():
    dset = MaterialsProjectDataset.from_devset(
        transforms=[
            transforms.PointCloudToGraphTransform("dgl", cutoff_dist=20.0),
            transforms.DistancesTransform(),
        ],
    )
    data = [dset.__getitem__(index) for index in range(2)]
    batch = dset.collate_fn(data)
    return batch


def test_mpnn_forward(test_batch):
    model = MPNN(atom_embedding_dim=128, encoder_only=True, node_out_dim=32)
    with torch.no_grad():
        g_z = model(test_batch)
    # 2 graphs, node_out_dim
    assert g_z.system_embedding.shape == (2, 32)
