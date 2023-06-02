import pytorch_lightning as pl
from ocpmodels.datasets import IS2REDataset, is2re_devset
from ocpmodels.datasets.transforms import COMShift
from ocpmodels.lightning.data_utils import PointCloudDataModule
from ocpmodels.models import GalaPotential
from ocpmodels.models.base import ScalarRegressionTask
from torch.nn import LazyBatchNorm1d, SiLU

pl.seed_everything(21616)

model_args = {
    "D_in": 200,
    "hidden_dim": 128,
    "merge_fun": "concat",
    "join_fun": "concat",
    "invariant_mode": "full",
    "covariant_mode": "full",
    "include_normalized_products": True,
    "invar_value_normalization": "momentum",
    "eqvar_value_normalization": "momentum_layer",
    "value_normalization": "layer",
    "score_normalization": "layer",
    "block_normalization": "layer",
    "equivariant_attention": False,
    "tied_attention": True,
    "encoder_only": False,
}


model = GalaPotential(**model_args)
task = ScalarRegressionTask(
    model,
    output_kwargs={"norm": LazyBatchNorm1d, "hidden_dim": 256, "activation": SiLU},
    lr=1e-3,
)

transforms = COMShift()

dm = PointCloudDataModule(
    train_path=is2re_devset,
    val_path=is2re_devset,
    dataset_class=IS2REDataset,
    batch_size=2,
    sample_size=12,
    num_workers=0,
    transforms=[transforms]
)

trainer = pl.Trainer(
    limit_train_batches=10,
    limit_val_batches=10,
    max_epochs=10,
    accelerator="cpu",
)

trainer.fit(task, datamodule=dm)