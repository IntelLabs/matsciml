import pytorch_lightning as pl

from matsciml.models.base import ScalarRegressionTask
from matsciml.lightning.data_utils import MatSciMLDataModule
from matsciml.datasets.transforms import PointCloudToGraphTransform

from matgl.models import M3GNet


# construct a scalar regression task with SchNet encoder
task = ScalarRegressionTask(
    encoder_class=M3GNet,
    encoder_kwargs={
        "encoder_only": True,
    },
    output_kwargs={"lazy": False, "input_dim": 64},
    task_keys=["band_gap"],
)
# Materials Project data needs the PointCloudToGraphTransform when using DGL.
dm = MatSciMLDataModule.from_devset(
    "MaterialsProjectDataset",
    dset_kwargs={
        "transforms": [
            PointCloudToGraphTransform(
                "dgl", cutoff_dist=20.0, node_keys=["pos", "atomic_numbers"]
            )
        ]
    },
    num_workers=0,
)

# run a quick training loop
trainer = pl.Trainer(fast_dev_run=10)
trainer.fit(task, datamodule=dm)
