import pytorch_lightning as pl
from torch.nn import LazyBatchNorm1d, SiLU

from ocpmodels.lightning.data_utils import MatSciMLDataModule
from ocpmodels.datasets.transforms import PointCloudToGraphTransform
from ocpmodels.models import GraphConvModel
from ocpmodels.models.base import ScalarRegressionTask

pl.seed_everything(21616)


model = GraphConvModel(100, 1, encoder_only=True)
task = ScalarRegressionTask(
    model,
    output_kwargs={"norm": LazyBatchNorm1d, "hidden_dim": 256, "activation": SiLU},
    lr=1e-3,
    task_keys=["energy_per_atom"],
)


dm = MatSciMLDataModule(
    "MaterialsProjectDataset",
    train_path="mp_data/base",
    dset_kwargs={"transforms": [PointCloudToGraphTransform("dgl", cutoff_dist=20.)]},
    val_split=0.2
    )

trainer = pl.Trainer(max_epochs=10, enable_checkpointing=False)

trainer.fit(task, datamodule=dm)
