from __future__ import annotations

from argparse import ArgumentParser

import pytorch_lightning as pl

from matsciml.datasets.utils import element_types
from matsciml.lightning.data_utils import MatSciMLDataModule
from matsciml.models import M3GNet
from matsciml.models.base import ScalarRegressionTask

parser = ArgumentParser()
parser.add_argument("--gpu", action="store_true", help="Use GPU for training or not.")
args = parser.parse_args()

if args.gpu == True:
    import torch

    torch.set_default_device("cuda")
    ACCELERATOR = "gpu"
else:
    ACCELERATOR = "cpu"


# construct a scalar regression task with SchNet encoder
task = ScalarRegressionTask(
    encoder_class=M3GNet,
    encoder_kwargs={
        "element_types": element_types(),
    },
    output_kwargs={"lazy": False, "input_dim": 64, "hidden_dim": 64},
    task_keys=["energy_total"],
)

dm = MatSciMLDataModule.from_devset(
    "M3GNomadDataset",
    num_workers=0,
    batch_size=4,
)

# run a quick training loop
trainer = pl.Trainer(fast_dev_run=10, accelerator=ACCELERATOR)
trainer.fit(task, datamodule=dm)
