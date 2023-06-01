
import pytorch_lightning as pl

from ocpmodels.lightning.data_utils import SyntheticPointGroupDataModule
from ocpmodels.models import GraphConvModel
from ocpmodels.models.base import CrystalSymmetryClassificationTask


dm = SyntheticPointGroupDataModule.from_devset()

task = CrystalSymmetryClassificationTask(
    encoder_class=GraphConvModel,
    encoder_kwargs={"atom_embedding_dim": 200, "out_dim": 1, "encoder_only": True}
)

trainer = pl.Trainer(fast_dev_run=100)
trainer.fit(task, datamodule=dm)
