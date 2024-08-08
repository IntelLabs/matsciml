from __future__ import annotations

import pytest
import numpy as np
from pytorch_lightning import Trainer

from matsciml.lightning.loss_scaling import LinearScalingSchedule
from matsciml.lightning.callbacks import LossScalingScheduler
from matsciml.models.pyg import EGNN
from matsciml.models import ScalarRegressionTask
from matsciml.lightning import MatSciMLDataModule
from matsciml.datasets.transforms import (
    PeriodicPropertiesTransform,
    PointCloudToGraphTransform,
)


@pytest.fixture
def task_and_dm():
    t = ScalarRegressionTask(
        encoder_class=EGNN,
        encoder_kwargs={"hidden_dim": 128, "output_dim": 64},
        task_keys=["energy"],
    )
    dm = MatSciMLDataModule.from_devset(
        "LiPSDataset",
        dset_kwargs={
            "transforms": [
                PeriodicPropertiesTransform(6.0, True),
                PointCloudToGraphTransform(
                    "pyg", node_keys=["pos", "atomic_numbers", "force"]
                ),
            ]
        },
    )
    return t, dm


@pytest.mark.parametrize("initial", (0.5, 1.5, 5.0))
@pytest.mark.parametrize("end", (2.0, 0.5, 100.0))
@pytest.mark.parametrize("num_steps", (10, 100, 5000))
def test_linear_schedule_without_trainer(initial, end, num_steps):
    sched = LinearScalingSchedule("test", initial, end)
    sched.set_grid(num_steps)
    rates = [sched.step() for _ in range(len(sched))]
    assert rates
    assert len(rates) == num_steps
    expected = np.linspace(initial, end, num_steps)
    assert np.allclose(np.array(rates), expected)


def test_linear_schedule_with_trainer(task_and_dm):
    """Tests that the linear schedule works under intended conditions."""
    task, dm = task_and_dm
    sched_callback = LossScalingScheduler(
        LinearScalingSchedule("energy", 1.0, 10.0, "step"),
    )
    trainer = Trainer(fast_dev_run=10, callbacks=sched_callback)
    trainer.fit(task, datamodule=dm)
    scheduler = sched_callback.schedules[0]
    # make sure that the scaling values are set correctly
    assert task.task_loss_scaling["energy"] != scheduler.initial_value
    assert task.task_loss_scaling["energy"] == scheduler.end_value


def test_linear_schedule_with_bad_key(task_and_dm):
    """Tests that the linear schedule works under intended conditions."""
    task, dm = task_and_dm
    sched_callback = LossScalingScheduler(
        LinearScalingSchedule("non-existent-key", 1.0, 10.0, "step"),
    )
    trainer = Trainer(fast_dev_run=10, callbacks=sched_callback)
    with pytest.raises(KeyError):
        trainer.fit(task, datamodule=dm)
