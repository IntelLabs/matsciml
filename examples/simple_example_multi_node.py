# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: MIT License

import pytorch_lightning as pl
import argparse
import torch
import os

from ocpmodels.datasets import devset_path
from ocpmodels.models import DimeNetPP, S2EFLitModule
from ocpmodels.lightning.data_utils import DGLDataModule

BATCH_SIZE = 16
NUM_WORKERS = 4
REGRESS_FORCES = False

# default model configuration for DimeNet++
model_config = {
    "emb_size": 128,
    "out_emb_size": 256,
    "int_emb_size": 64,
    "basis_emb_size": 8,
    "num_blocks": 2,
    "num_spherical": 7,
    "num_radial": 6,
    "cutoff": 10.0,
    "envelope_exponent": 5.0,
    "activation": torch.nn.SiLU,
}


def main(args):
    # use default settings for DimeNet++
    dpp = DimeNetPP(**model_config)

    # use the GNN in the LitModule for all the logging, loss computation, etc.
    model = S2EFLitModule(dpp, regress_forces=REGRESS_FORCES, lr=1e-3, gamma=0.1)

    data_module = DGLDataModule(devset_path)

    trainer = pl.Trainer(
        accelerator="gpu",
        strategy="ddp",
        devices=args.num_devices,
        num_nodes=args.world_size,
    )

    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    doc = """
    Basic example showing how to launch distributed multi-node training. Each node must
    contain this repositories code, datasets, and additional required packages. After 
    configuring each node, note down the required parameters, namely:

    MASTER_PORT - required; has to be a free port on machine with NODE_RANK 0
    MASTER_ADDR - required; address of NODE_RANK 0 node
    WORLD_SIZE - required; how many nodes are in the cluster
    NODE_RANK - required; id of the node in the cluster
    NUM_DEVICES - required; number of devices per node.

    Once ready, this scrip must be launched on each node separately with the required 
    arguments. For example, if using two nodes, with two gpu's available, the script
    should be launched as:
    (Node 0)
    python simple_example_multi_node.py --master-addr 10.42.72.12 --master-port 12345 --world-size 2 --node-rank 0 --num-devices 2
    (Node 1)
    python simple_example_multi_node.py --master-addr 10.42.72.12 --master-port 12345 --world-size 2 --node-rank 1 --num-devices 2
    """
    parser = argparse.ArgumentParser(description=doc)
    parser.add_argument("--master-addr", required=True, help="IP address of master node")
    parser.add_argument("--master-port", required=True, help="Open port on master node")
    parser.add_argument("--world-size", required=True, help="Total number of nodes")
    parser.add_argument("--node-rank", required=True, help="Rank of current node")
    parser.add_argument("--num-devices", required=True, help="Number of devices per node")
    args = parser.parse_args()

    os.environ["MASTER_ADDR"] = args.master_addr
    os.environ["MASTER_PORT"] = args.master_port
    os.environ["WORLD_SIZE"] = args.world_size
    os.environ["NODE_RANK"] = args.node_rank

    main(args)
