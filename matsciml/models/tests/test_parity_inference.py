from __future__ import annotations

import shutil
import json
from pathlib import Path

import pytest
import pytorch_lightning as pl

from matsciml.models.inference import ParityInferenceTask
from matsciml.models.base import ScalarRegressionTask
from matsciml.models.pyg import EGNN
from matsciml.lightning import MatSciMLDataModule
from matsciml.datasets.transforms import (
    PeriodicPropertiesTransform,
    PointCloudToGraphTransform,
)


@pytest.mark.slow
@pytest.mark.parametrize(
    "dset_params",
    [
        (
            "MaterialsProjectDataset",
            [
                "efermi",
            ],
        ),
        (
            "LiPSDataset",
            [
                "energy",
            ],
        ),
        ("OQMDDataset", ["stability", "band_gap"]),
    ],
)
def test_parity_inference_workflow(dset_params):
    dataset_name, keys = dset_params
    dm = MatSciMLDataModule.from_devset(
        dataset_name,
        dset_kwargs={
            "transforms": [
                PeriodicPropertiesTransform(6.0, True),
                PointCloudToGraphTransform("pyg"),
            ]
        },
        batch_size=8,
    )
    task = ScalarRegressionTask(
        encoder_class=EGNN,
        encoder_kwargs={"hidden_dim": 16, "output_dim": 16, "num_conv": 2},
        output_kwargs={"hidden_dim": 16},
        task_keys=keys,
    )
    # train the model briefly to initialize output heads
    trainer = pl.Trainer(max_epochs=1, limit_train_batches=10, limit_val_batches=0)
    trainer.fit(task, dm)
    # now do the inference part
    wrapper = ParityInferenceTask(task)
    trainer.predict(wrapper, datamodule=dm)
    assert trainer.log_dir is not None
    log_dir = Path(trainer.log_dir)
    assert log_dir.exists()
    # open the result and make sure it's not empty
    with open(log_dir.joinpath("inference_data.json"), "r") as read_file:
        data = json.load(read_file)
    assert len(data) != 0
    assert sorted(list(data.keys())) == sorted(task.task_keys)
    # make sure there are actually predictions and targets available
    for subdict in data.values():
        assert len(subdict["predictions"]) == len(subdict["targets"])
    shutil.rmtree("lightning_logs", ignore_errors=True)
