import pytorch_lightning as pl

from ocpmodels.lightning.data_utils import MatSciMLDataModule
from ocpmodels.models import GraphConvModel
from ocpmodels.models.base import ScalarRegressionTask
from ocpmodels.datasets.transforms import PointCloudToGraphTransform


# configure a simple model for testing
model = GraphConvModel(100, 1, encoder_only=True)
task = ScalarRegressionTask(model, task_keys=["band_gap"])

# configure materials project from devset
dm = MatSciMLDataModule.from_devset(
    "MaterialsProjectDataset",
    dset_kwargs={"transforms": [PointCloudToGraphTransform("dgl", cutoff_dist=20.0)]},
)

# run 10 steps for funsies
trainer = pl.Trainer(fast_dev_run=10, enable_checkpointing=False, logger=False)

trainer.fit(task, datamodule=dm)
