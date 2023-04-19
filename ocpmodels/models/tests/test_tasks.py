
import pytorch_lightning as pl

from ocpmodels.models.base import ForceRegressionTask
from ocpmodels.models import GraphConvModel
from ocpmodels.lightning.data_utils import MaterialsProjectDataModule, S2EFDGLDataModule

pl.seed_everything(2156161)


def test_regression_devset():
    devset = MaterialsProjectDataModule.from_devset()


def test_force_regression():
    devset = S2EFDGLDataModule.from_devset()
    model = GraphConvModel(100, 1, encoder_only=True)
    task = ForceRegressionTask(model)
    trainer = pl.Trainer(max_steps=5, logger=False, enable_checkpointing=False)
    trainer.fit(task, datamodule=devset)
    # make sure losses are tracked
    for key in ["energy", "force"]:
        assert f"train_{key}_step" in trainer.logged_metrics

