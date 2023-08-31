import pytorch_lightning as pl
from torch.nn import LazyBatchNorm1d, SiLU

from ocpmodels.lightning.data_utils import MaterialsProjectDataModule
from ocpmodels.datasets.materials_project import DGLMaterialsProjectDataset
from ocpmodels.models import GraphConvModel
from ocpmodels.models.base import CrystalSymmetryClassificationTask

pl.seed_everything(21616)


model = GraphConvModel(100, 1, encoder_only=True)
task = CrystalSymmetryClassificationTask(
    model,
    output_kwargs={"norm": LazyBatchNorm1d, "hidden_dim": 256, "activation": SiLU},
    lr=1e-3,
)

# the base set is required because the devset does not contain symmetry labels
dm = MaterialsProjectDataModule(
    dataset=DGLMaterialsProjectDataset("mp_data/base", cutoff_dist=10.0),
    val_split=0.2,
)

trainer = pl.Trainer(max_epochs=10, enable_checkpointing=False)

trainer.fit(task, datamodule=dm)
