try:
    import oneccl_bindings_for_pytorch

    _has_ccl = True
except ImportError:
    _has_ccl = False

import pytest
import pytorch_lightning as pl
from ocpmodels.lightning.data_utils import (IS2REDGLDataModule,
                                            MaterialsProjectDataModule)
from ocpmodels.lightning.ddp import MPIEnvironment
from ocpmodels.models import GraphConvModel
from ocpmodels.models.base import ScalarRegressionTask
from pytorch_lightning.strategies.ddp import DDPStrategy


@pytest.mark.skipif(not _has_ccl, reason="No working oneCCL installation.")
def test_ccl_is2re_ddp():
    devset = IS2REDGLDataModule.from_devset()

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
    devset = MaterialsProjectDataModule.from_devset(graphs=True)

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

