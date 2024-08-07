from __future__ import annotations

import pytest
import numpy as np

from matsciml.lightning.loss_scaling import LinearScalingSchedule
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
