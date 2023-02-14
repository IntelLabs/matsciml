from argparse import ArgumentParser

from ocpmodels.models import OE62LitModule, MEGNet
from ocpmodels.lightning.data_utils import OE62DGLDataModule
from ocpmodels.datasets import transforms as t
from ocpmodels.lightning.ddp import IntelMPIEnvironment

import pytorch_lightning as pl
from pytorch_lightning.strategies.ddp import DDPStrategy

"""
Example script to train MegNet on OE62 using distributed data parallelism.

This script is designed to run on CPU, and will need minor tweaks to work
on GPUs (namely in the `Trainer` definition, add `accelerator="gpu"` and
the corresponding number of workers).
"""

parser = ArgumentParser()
parser.add_argument(
    "-ppn",
    "--num_procs_per_node",
    default=1,
    type=int,
    help="Number of workers per node.",
)
parser.add_argument(
    "-nn", "--num_nodes", default=1, type=int, help="Number of per nodes in the pool."
)

args = parser.parse_args()

ppn = args.num_procs_per_node
nn = args.num_nodes

assert ppn >= 1, f"Number of procs per node must be at least 1"
assert nn >= 1, f"Number of nodes must be at least"
world_size = ppn * nn


dm = OE62DGLDataModule(
    "/data/datasets/oe62",
    batch_size=64,
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

# scale learning rate by square root of world size, normalize the bandgap by
# the mean and td of the whole dataset
lit_module = OE62LitModule(
    model,
    lr=1e-3 * world_size**0.5,
    gamma=0.9,
    normalize_kwargs={
        "bandgap_mean": 3.1578752994537354,
        "bandgap_std": 0.8467098474502563,
    },
)

# initialize DDP settings
env = IntelMPIEnvironment()
ddp = DDPStrategy(
    cluster_environment=env, process_group_backend="mpi", find_unused_parameters=False
)

trainer = pl.Trainer(
    max_epochs=1,
    num_nodes=nn,
    devices=ppn,
    strategy=ddp,
)
trainer.fit(lit_module, datamodule=dm)
