
import pytorch_lightning as pl

from ocpmodels.lightning.data_utils import MaterialsProjectDataModule
from ocpmodels.datasets.materials_project import MaterialsProjectDataset, DGLMaterialsProjectDataset
from ocpmodels.models.base import MultiTaskLitModule, ScalarRegressionTask, BinaryClassificationTask
from ocpmodels.models import GraphConvModel

pl.seed_everything(1616)

dset = DGLMaterialsProjectDataset("../materials_project/mp_data/base")
dm = MaterialsProjectDataModule(dset, batch_size=16)

model = GraphConvModel(100, 1, encoder_only=True)
r = ScalarRegressionTask(model, lr=1e-3)
c = BinaryClassificationTask(model, lr=1e-3)

task = MultiTaskLitModule(
    ("MaterialsProjectDataset", r),
    ("MaterialsProjectDataset", c)
)

trainer = pl.Trainer(max_steps=10, logger=False, enable_checkpointing=False)
trainer.fit(task, datamodule=dm)
