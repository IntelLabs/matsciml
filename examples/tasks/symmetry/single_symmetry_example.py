import pytorch_lightning as pl

from ocpmodels.lightning.data_utils import MatSciMLDataModule
from ocpmodels.datasets.transforms import PointCloudToGraphTransform
from ocpmodels.models import GraphConvModel
from ocpmodels.models.base import CrystalSymmetryClassificationTask


dm = MatSciMLDataModule.from_devset(
    "SyntheticPointGroupDataset",
    dset_kwargs={
        "transforms": [
            PointCloudToGraphTransform("dgl", cutoff_dist=15.0, node_keys=[])
        ]
    },
)

task = CrystalSymmetryClassificationTask(
    encoder_class=GraphConvModel,
    encoder_kwargs={"atom_embedding_dim": 128, "out_dim": 1, "encoder_only": True},
)

trainer = pl.Trainer(fast_dev_run=1000)
trainer.fit(task, datamodule=dm)
