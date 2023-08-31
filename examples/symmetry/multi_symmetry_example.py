import pytorch_lightning as pl

from matsciml.lightning.data_utils import MultiDataModule
from matsciml.datasets.symmetry import DGLSyntheticPointGroupDataset, symmetry_devset
from matsciml.datasets.materials_project import (
    DGLMaterialsProjectDataset,
    materialsproject_devset,
)
from matsciml.datasets.multi_dataset import MultiDataset
from matsciml.models import GraphConvModel
from matsciml.models.base import (
    CrystalSymmetryClassificationTask,
    MultiTaskLitModule,
    ScalarRegressionTask,
)

dm = MultiDataModule(
    train_dataset=MultiDataset(
        [
            DGLSyntheticPointGroupDataset(symmetry_devset),
            DGLMaterialsProjectDataset(materialsproject_devset),
        ]
    ),
    batch_size=16,
)

sym_task = CrystalSymmetryClassificationTask(
    encoder_class=GraphConvModel,
    encoder_kwargs={"atom_embedding_dim": 200, "out_dim": 1, "encoder_only": True},
)
reg_task = ScalarRegressionTask(
    encoder_class=GraphConvModel,
    encoder_kwargs={"atom_embedding_dim": 200, "out_dim": 1, "encoder_only": True},
    task_keys=["band_gap"],
)

task = MultiTaskLitModule(
    ("DGLSyntheticPointGroupDataset", sym_task),
    ("DGLMaterialsProjectDataset", reg_task),
)

trainer = pl.Trainer(max_epochs=1, log_every_n_steps=5)
trainer.fit(task, datamodule=dm)
