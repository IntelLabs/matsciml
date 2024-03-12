from __future__ import annotations

from matsciml.common.packages import package_registry

if package_registry["ccl"]:
    _has_ccl = True
else:
    _has_ccl = False

import pytest
import pytorch_lightning as pl
from torch import distributed as dist

from matsciml.datasets import transforms
from matsciml.datasets.materials_project import MaterialsProjectDataset
from matsciml.models import GraphConvModel
from matsciml.models.base import ScalarRegressionTask


@pytest.mark.skipif(not _has_ccl, reason="No working oneCCL installation.")
@pytest.mark.skipif(not dist.is_initialized(), reason="Distributed is not initilaized.")
@pytest.mark.distributed
def test_ccl_is2re_ddp():
    devset = MaterialsProjectDataset.from_devset(
        transforms=[transforms.PointCloudToGraphTransform("dgl", cutoff_dist=20.0)],
    )

    model = GraphConvModel(100, 1, encoder_only=True)
    task = ScalarRegressionTask(model, lr=1e-3)

    trainer = pl.Trainer(
        max_epochs=1,
        limit_train_batches=10,
        enable_checkpointing=False,
        logger=False,
        strategy="ddp_with_ccl",
        devices=2,
    )
    trainer.fit(task, datamodule=devset)


@pytest.mark.skipif(not _has_ccl, reason="No working oneCCL installation.")
@pytest.mark.skipif(not dist.is_initialized(), reason="Distributed is not initilaized.")
@pytest.mark.distributed
def test_ccl_materials_project():
    devset = MaterialsProjectDataset.from_devset(
        transforms=[transforms.PointCloudToGraphTransform("dgl", cutoff_dist=20.0)],
    )

    model = GraphConvModel(100, 1, encoder_only=True)
    task = ScalarRegressionTask(model, lr=1e-3)

    trainer = pl.Trainer(
        max_epochs=1,
        limit_train_batches=10,
        enable_checkpointing=False,
        logger=False,
        strategy="ddp_with_ccl",
        devices=2,
    )
    trainer.fit(task, datamodule=devset)
