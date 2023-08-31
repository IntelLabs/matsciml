import pytorch_lightning as pl
from torch.nn import LayerNorm, SiLU

from matsciml.lightning.data_utils import MatSciMLDataModule
from matsciml.datasets.transforms import PointCloudToGraphTransform
from matsciml.models import GraphConvModel
from matsciml.models.base import ScalarRegressionTask

pl.seed_everything(21616)


model = GraphConvModel(100, 1, encoder_only=True)
task = ScalarRegressionTask(
    model,
    output_kwargs={
        "norm": LayerNorm(128),
        "hidden_dim": 128,
        "activation": SiLU,
        "lazy": False,
        "input_dim": 128,
    },
    lr=1e-3,
    task_keys=["energy_per_atom"],
)


dm = MatSciMLDataModule(
    "MaterialsProjectDataset",
    train_path="mp_data/base",
    dset_kwargs={"transforms": [PointCloudToGraphTransform("dgl", cutoff_dist=20.0)]},
    val_split=0.2,
)

trainer = pl.Trainer(max_epochs=10, enable_checkpointing=False)

trainer.fit(task, datamodule=dm)
