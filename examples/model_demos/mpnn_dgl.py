import pytorch_lightning as pl

from ocpmodels.models import MPNN
from ocpmodels.models.base import ScalarRegressionTask
from ocpmodels.lightning.data_utils import IS2REDGLDataModule
from ocpmodels.datasets.transforms import DistancesTransform


# construct a scalar regression task with SchNet encoder
task = ScalarRegressionTask(
    encoder_class=MPNN,
    encoder_kwargs={"encoder_only": True, "node_embedding_dim": 128, "node_out_dim": 256},
    task_keys=["energy_relaxed"],
)
# SchNet uses RBFs, and expects edge features corresponding to atom-atom distances
dm = IS2REDGLDataModule.from_devset(transforms=[DistancesTransform()])

# run a quick training loop
trainer = pl.Trainer(fast_dev_run=10)
trainer.fit(task, datamodule=dm)
