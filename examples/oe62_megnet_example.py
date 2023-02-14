from ocpmodels.models import OE62LitModule, MEGNet
from ocpmodels.lightning.data_utils import OE62DGLDataModule
from ocpmodels.datasets import transforms as t

import pytorch_lightning as pl

"""
Bare minimum script that demonstrates running MegNet with the OE62 dataset.
"""

dm = OE62DGLDataModule(
    "/data/datasets/oe62",
    batch_size=32,
    num_workers=4,
    transforms=[t.DistancesTransform(), t.GraphVariablesTransform()],
)

model = MEGNet(
    edge_feat_dim=2,
    node_feat_dim=1,
    graph_feat_dim=9,
    num_blocks=4,
    hiddens=[128, 64, 64],
    conv_hiddens=[128, 128, 64],
    s2s_num_layers=5,
    s2s_num_iters=4,
    output_hiddens=[64, 16],
    is_classification=False,
    dropout=0.1,
    num_atom_embedding=100,
)

lit_module = OE62LitModule(
    model,
    lr=1e-3,
    gamma=0.9,
    normalize_kwargs={"bandgap_mean": 0.0, "bandgap_std": 1.0},
)

trainer = pl.Trainer(max_epochs=1)
trainer.fit(lit_module, datamodule=dm)
