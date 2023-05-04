import pytest

import pytorch_lightning as pl

from ocpmodels.datasets.multi_dataset import MultiDataset
from ocpmodels.datasets import IS2REDataset, is2re_devset, S2EFDataset, s2ef_devset
from ocpmodels.lightning.data_utils import MultiDataModule

from ocpmodels.models.base import (
    MultiTaskLitModule,
    ForceRegressionTask,
    ScalarRegressionTask,
)
from ocpmodels.models import GraphConvModel


@pytest.fixture
def is2re_s2ef() -> MultiDataModule:
    dm = MultiDataModule(
        train_dataset=MultiDataset(
            [IS2REDataset(is2re_devset), S2EFDataset(s2ef_devset)]
        ),
        batch_size=16,
    )
    return dm


@pytest.mark.dependency()
def test_target_keys(is2re_s2ef):
    dm = is2re_s2ef
    keys = dm.target_keys
    assert keys == {
        "IS2REDataset": {"regression": ["energy_init", "energy_relaxed"]},
        "S2EFDataset": {"regression": ["energy", "force"]},
    }


@pytest.mark.dependency(depends=["test_target_keys"])
def test_multitask_init(is2re_s2ef):
    dm = is2re_s2ef

    encoder = GraphConvModel(100, 1, encoder_only=True)
    is2re = ScalarRegressionTask(encoder)
    s2ef = ForceRegressionTask(encoder)

    # pass task keys to make sure output heads are created
    task = MultiTaskLitModule(
        ("IS2REDataset", is2re), ("S2EFDataset", s2ef), task_keys=dm.target_keys
    )
    # make sure we have output heads
    assert hasattr(is2re, "output_heads")
    assert hasattr(s2ef, "output_heads")
    assert "energy_init" in is2re.output_heads
    assert "energy" in s2ef.output_heads


@pytest.mark.dependency(depends=["test_multitask_init"])
def test_multitask_end2end(is2re_s2ef):
    dm = is2re_s2ef

    encoder = GraphConvModel(100, 1, encoder_only=True)
    is2re = ScalarRegressionTask(encoder)
    s2ef = ForceRegressionTask(encoder)

    task = MultiTaskLitModule(
        ("IS2REDataset", is2re), ("S2EFDataset", s2ef), task_keys=dm.target_keys
    )
    trainer = pl.Trainer(logger=False, enable_checkpointing=False, fast_dev_run=1)
    trainer.fit(task, datamodule=dm)

