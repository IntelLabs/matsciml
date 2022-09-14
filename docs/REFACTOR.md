
# Open Cataylst Project

**PyTorch Lightning Edition**

## Introduction

## Installation

A minimum working `conda` environment specification is included in `pl-env.yml`. It should guarantee
an up and running environment, albeit with a PyTorch build relying on a suboptimal CPU-only version
through the PyTorch anaconda channel. This is to ensure that the environment builds successfully in
a wider range of situations, and defers the building of a better (e.g. CUDA-enabled, oneDNN, etc.)
version to the user and specific to what hardware is available.

It's also recommended to install the `ocpmodels` as a `pip` package, which will allow your workflow
to be agnostic of the repository filestructure: once the `conda` environment is built, run:

```bash
pip install -e .
```

to install `ocpmodels` as a softlink, whereby changes in the `ocpmodels` codebase will be reflected
at runtime for your environment.

---

## Major changes from original implementation

The main purpose of this refactor was to port MLOps functionality&mdash;including but not limited to
logging, distributed computing, accelerator usage, CLI&mdash;to PyTorch Lightning, which minimizes
the code complexity of `ocpmodels`. The design philosophy with PL in mind is to abstract when possible
(e.g. model definitions separated from training), and when something more specialized is required,
write it in native PyTorch&mdash;for example, a CLI that works for all models versus a training script
for performance tuning in HPC environments.

This section will go over some of the specific details. It's recommended that the reader gains an
understanding of base OCP before reading this, as far as to have a basic understanding of how the
package/project is laid out.

### Task abstraction

OCP provides two scientific tasks: predicting energy and forces from structures ("S2EF"), and mapping
unrelaxed structures to relaxed ones ("IS2RE"). The original implementation includes abstractions for
these two tasks throughout the repository, namely in the form of different trainers.

In this refactor, we replace all of the original definitions in the `ocpmodels/trainers` modules with
PyTorch Lightning `LightningModule`s: we have a base class, `OCPLitModule`, which implements some core
boilerplate code such as handling normalizers and a placeholder for an abstract graph neural network
model. We then implement two separate subclasses for the two OCP tasks: a `S2EFLightningModule`, and an
`IS2RELightningModule`; these two modules define the overall workflow and computation needed to train,
test, and predict for these tasks. These are implemented in `ocpmodels/models/base.py`.

At a high level, the former takes an abstract GNN model that reduces
a molecular graph into a scalar energy prediction&mdash;the `regress_forces` flag will automatically
handle force computation (as the derivative of the energy with respect to atom positions), such that
the `forward` implements the energy/force logic without having to duplicate force computation for other
models. The `training_step` method subsequently defines the training logic, including the second backward
pass necessary for the force computation, takes care of logging and optimizer step[^1]. We define a generic
`_compute_losses` function which, given a batch of graphs, will compute the energy and force (if requested)
and return a dictionary of loss values. Currently, only the `S2EFLightningModule` is functional and tested.

To help streamline the model development process, we have previously mentioned these `LightningModules` comprise
an abstract graph neural network, which regresses the energy. To help with the abstraction, we have defined a
base `AbstractTask`, from which we subclass as `AbstractEnergyModel` and `AbstractIS2REModel`. If you are
interested in developing/creating a new model architecture for use with OCP, you can simply create a model that
inherits from one or the other. A minimal example is given in `ocpmodels/models/gcn.py`, which defines a `GCNConv`
model that inherits from `AbstractEnergyModel`, with the expected output of a scalar energy value. This abstraction
allows for one to easily plug and play new models without changing the rest of the pipeline: simply replace the
`gnn` in a `LightningModule` for a new one, and the rest of the behavior/logic should stay consistent.

#### Current list of DGL ported models

(TODO add credits and citations)

- DimeNet++
- EGNN
- GAANet
- Basic GCN

### Data

Data loading, splits, and load balancing are implemented as a base `GraphDataModule` class in `ocpmodels.lightning.data`.
This class takes abstract dataset and loader objects to support either PyG or DGL frameworks: to use either framework,
call their specific subclasses (e.g. `DGLDataModule`).

---

## Quick reference

### Running the workload

If `ocpmodels` has been `pip` installed, the CLI is the main recommended route for running the workload; see
the CLI usage section for details.

### MultiGPU

In the `trainer` configuration, specify `num_gpus`. For now, you will also need to specify `num_gpus` in the
`data.init_args` as well; make sure the two are consistent.

### Distributed data parallel

In the `trainer` configuration, specify `ddp` for the `strategy` key. The number of processes can be controlled
with the `devices` key, which refers to the number of CPU or general accelerators.

### Distributed backend

In older versions of PyTorch Lightning, the `PL_TORCH_DISTRIBUTED_BACKEND` environment variable sets the
distributed backend; can be `gloo`, `mpi`, `nccl`. 

For optimal configuration on CPU, it is recommended to use Intel MPI, in addition to configuring for
NUMA-binding in distributed settings. To this end, there are a set of specialized PyTorch Lightning
DDP strategies and Environments: the latter in particular will parse environment variables set by
Intel MPI and enable its utilization.

### Defunct components

Currently, most modules are not actively being used and have been kept from the earlier implementation to defer
review over porting functionality. The modules that have received changes (of varying degree) are:

- `ocpmodels.lightning` and submodules
- `ocpmodels.models.base`, and other submodules that implement new models
- `ocpmodels.datasets` and `ocpmodels.preprocessing` which contain changes for DGL data structure support

The remaining modules are unlikely to be touched, neither actively (as a user) and as a developer.

### Logging

All model logging is handled by PyTorch Lightning, and can be found in `ocpmodels/models/base.py` as part of the
`LightningModule` train/validation/test steps. The data will be logged to `tqdm` progress bar, as well as any
selected logger (defaults to Tensorboard).

### Defining new models

Create a new module under `ocpmodels.models`, and inhert from `AbstractEnergyModel`. As a quick illustration:

```python
from torch import nn
from ocpmodels.models.base import AbstractEnergyModel

class MyModel(AbstractEnergyModel):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        # define your model
        ...

    # just implement the forward method; expected output is
    # a 2D tensor of shape [N, 1] for N minibatch size
    def forward(self, graph: DGLGraph) -> torch.Tensor:
        ...

```

To use the model in training, we recommend using a YAML configuration with the CLI (see next section).

### CLI usage

The CLI is relatively straightforward, and users are pointed to the [PyTorch Lightning docs](https://pytorch-lightning.readthedocs.io/en/latest/cli/lightning_cli.html)
for general information on how the interface works. Specific to OCP, we define CLIs for the two tasks, which can be
launched via:

```bash
python -m ocpmodels.lightning.s2ef <subcommand> --config test.yml
```

Subcommand can be replaced by one of the `Trainer` class commands, such as `fit`, `test`, `predict`, and `tune`. The
`--config` option can be specified multiple times, and point to YAML configuration files that define the components
of the whole workflow. Below is an example that will piece everything together:

```yaml
model:
  gnn: 
    class_path: ocpmodels.models.gcn.GraphConvModel
    init_args:
      atom_embedding_dim: 128
      out_dim: 32
      num_blocks: 3
      num_fc_layers: 3
      activation: torch.nn.SiLU
      readout: dgl.nn.pytorch.SumPooling
  regress_forces: False
  normalize_kwargs:
    target_mean: -0.7554450631141663
    target_std: 2.887317180633545
    grad_target_mean: 0.0
    grad_target_std: 2.887317180633545
data:
  class_path: ocpmodels.lightning.data.DGLDataModule
  init_args:
    train_path: "data/s2ef/200k/train"
    batch_size: 64
    num_workers: 1
optimizer:
  class_path: torch.optim.Adam
  init_args:
    lr: 0.0005
lr_scheduler:
  class_path: torch.optim.lr_scheduler.ExponentialLR
  init_args:
    gamma: 0.1
trainer:
  max_steps: 3
  limit_val_batches: 0
  limit_test_batches: 0
  limit_predict_batches: 0
  accelerator: "cpu"
  num_nodes: 1
  devices: 1
  strategy: null
seed_everything: 42
```

The enticing part of this configuration is that models can be readily configured and composed
simply by replacing the `gnn` key within `model`&mdash;replace it with another abstract GNN,
and the rest of the pipeline works without any other changes (at least that's the hope!). Other
notable components are `data`, which points to the `DGLDataModule` that handles the splits and loaders,
`optimizer` and `lr_scheduler` which allows control over these components without needing to overwrite
`configure_optimizers` within the `LightningModules`, and the `trainer` key which controls the various
aspects of the training loop, such as the number of steps/epochs, strategies for distributed training,
and so on. 

One important thing to note here is the `class_path` and `init_args` pattern: this is used by `jsonargparse`
to create arbitrary objects that are fed as arguments. Notably this is used for the model definition,
optimizer/scheduler, and data modules, however will work for other arguments as well such as `trainer.logger`
which can be user-defined, or other experiment tracking services like Weights and Biases.

---

[^1]: For double gradient computation, we have to override `automatic_optimization` for `LightningModules`, which
is why we instead call `manual_backward`.
