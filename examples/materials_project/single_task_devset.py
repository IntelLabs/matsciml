
import pytorch_lightning as pl

from ocpmodels.lightning.data_utils import MaterialsProjectDataModule
from ocpmodels.models import GraphConvModel
from ocpmodels.models.base import ScalarRegressionTask


model = GraphConvModel(100, 1, encoder_only=True)
task = ScalarRegressionTask(model)

dm = MaterialsProjectDataModule.from_devset()

trainer = pl.Trainer(max_steps=10, enable_checkpointing=False, logger=False)

trainer.fit(task, datamodule=dm)

