import pytorch_lightning as pl
from torch.nn import LazyBatchNorm1d, SiLU

from ocpmodels.lightning.data_utils import MaterialsProjectDataModule
from ocpmodels.datasets.materials_project import DGLMaterialsProjectDataset
from ocpmodels.models import GraphConvModel
from ocpmodels.models.base import ScalarRegressionTask, BinaryClassificationTask
from ocpmodels.lightning import callbacks

pl.seed_everything(21616)


model = GraphConvModel(100, 1, encoder_only=True)
task = BinaryClassificationTask(
    model,
    output_kwargs={"norm": LazyBatchNorm1d, "hidden_dim": 256, "activation": SiLU},
    lr=1e-3,
)

dm = MaterialsProjectDataModule(
    DGLMaterialsProjectDataset("mp_data/base", cutoff_dist=10.0), val_split=0.2, batch_size=64
)
dm.setup()
loader = dm.train_dataloader()
batch = next(iter(loader))
# import pdb; pdb.set_trace()

trainer = pl.Trainer(max_epochs=10, enable_checkpointing=False, accelerator="gpu", devices=1)

trainer.fit(task, datamodule=dm)
