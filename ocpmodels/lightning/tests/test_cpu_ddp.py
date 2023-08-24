try:
    import oneccl_bindings_for_pytorch

    _has_ccl = True
except ImportError:
    _has_ccl = False

import pytest
import pytorch_lightning as pl
from ocpmodels.datasets.materials_project import MaterialsProjectDataset
from ocpmodels.lightning.ddp import MPIEnvironment
from ocpmodels.models import GraphConvModel
from ocpmodels.models.base import ScalarRegressionTask
from pytorch_lightning.strategies.ddp import DDPStrategy
from ocpmodels.datasets import transforms


@pytest.mark.skipif(not _has_ccl, reason="No working oneCCL installation.")
def test_ccl_is2re_ddp():
    devset = MaterialsProjectDataset.from_devset(
        transforms=[transforms.PointCloudToGraphTransform("dgl", cutoff_dist=20.0)]
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
def test_ccl_materials_project():
    devset = MaterialsProjectDataset.from_devset(
        transforms=[transforms.PointCloudToGraphTransform("dgl", cutoff_dist=20.0)]
    )

    model = GraphConvModel(100, 1, encoder_only=True)
    task = ScalarRegressionTask(model, lr=1e-3)

    env = MPIEnvironment()
    ddp = DDPStrategy(cluster_environment=env, process_group_backend="ccl")

    trainer = pl.Trainer(
        max_epochs=1,
        limit_train_batches=10,
        enable_checkpointing=False,
        logger=False,
        strategy="ddp_with_ccl",
        devices=2,
    )
    trainer.fit(task, datamodule=devset)
