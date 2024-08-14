# Experiments in Open MatSciML Toolkit

Experimental workflows may be time consuming, repetitive, and complex to set up. Additionally, pytorch-lightning based cli utilities may not be able to handle specific use cases such as multi-data or multi-task training in matsciml. The experiments module of MatSciML is meant to loosely mirror the functionality of the pytorch lightning cli while allowing more flexibility in setting up complex experiments. Yaml files define the module parameters, and specific arguments may be change via the command line if desired. A single command is used to launch training runs which take out the complexity of writing up new script for each experiment type.

## Config Files
Yaml files dictate how models, datasets, tasks and trainers are set up. A default set of config files is provided under `./experiments/configs`, however when setting up new experiments, it is recommended to create a folder of your own to better track your own experiment designs. Using the cli parameters `-d` `-m`, `-t` and `-e`, you can specify a path to a directory or specific file for datasets, models, trainer, and experiment configuration. The varying yaml files and their contents are explained below. An example of running a simple experimental using the predefined yaml files may look like this:

```python
python experiments/training_script.py -e experiments/experiment_config.yaml -t experiments/configs/trainer.yaml -m ./experiments/models -d ./experiments/datasets/oqmd.yaml --debug
```

Note that a combination of full yaml file paths and directory paths are used. In the case of models configs (`-m`) a full path is specified, meaning all yaml files contained in that directory will accessible from the experiment config. In the case of multidata training, it is only possible to point to a directory of datasets as multiple will need to be configured.

## Experiment Config
The starting point of defining an experiment is the experiment config. This is a yaml file that lays out what model, dataset(s), and task(s) will be used during training. An example config for single task training yaml (`single_task.yaml`) look like this:
```yaml
model: egnn_dgl
dataset:
  oqmd:
    scalar_regression:
      - energy
```

### Checkpoint Loading
Pretrained model checkpoints may be loaded for use in downstream tasks. Models can be loaded and used *as-is*, or only the encoder may be used.

To load a checkpoint, add the `load_weights` field to the experiment config:
```yaml
model: egnn_dgl
dataset:
  oqmd:
    - task: ScalarRegressionTask
      targets:
       - energy
load_weights:
   method: checkpoint
   type: local
   path: ./path/to/checkpoint
```
* `method` specifies whether to use the model *as-is* (`checkpoint`), or *encoder-only* (`pretrained`) in the checkpoint.
* `type` specifies where to load the checkpoint from (`local`, or `wandb`).
* `path` points to the location of the checkpoint. WandB checkpoints may be specified by pointing to the model artifact, typically specified by: `entity-name/project-name/model-version:number`


In general, and experiment may the be launched by running:
`python experiments/training_script.py --experiment_config ./experiments/configs/single_task.yaml`


* The trainer used defaults to the config in `./experiments/configs/trainer.yaml`.
* The `model` field points to a specific `model.yaml` file. Default model configs are in `./experiments/configs/models`.
* The `dataset` field is a dictionary specifying which datasets to use, as well as which tasks are associated with the parent dataset. Default datasets are in `./experiments/configs/datasets`.
    * Tasks are referred to by their class name:
    ```python
    ScalarRegressionTask
    ForceRegressionTask
    BinaryClassificationTask
    CrystalSymmetryClassificationTask
    GradFreeForceRegressionTask
    ```
* A dataset may contain more than one task (single data, multi task learning)
* Multiple datasets can be provided, each containing their own tasks (multi data, multi task learning)
* For a list of available datasets, tasks, and models run `python training_script.py --options`.

## Trainer Config
The training config contains a few sections used for defining how experiments will be run. The debug tag is used to set parameters that should be used when debugging an experimental setup, or when working through bugs in setting up a new model or dataset. These parameters are helpful for doing quick end-to-end runs to make sure the pipeline is functional. The experiment tag is used to define parameters for the full experiment runs. Finally the generic tag used to define parameters used regardless of going through a debug or full experimental run.

In addition to the experiment types, any other parameters to be used with the pytorch lightning `Trainer` should be added here. In the example `trainer.yaml`, there are callbacks and a logger. These objects are set up by adding their `class_path` as well as any `init_args` they expect.
```yaml
generic:
  min_epochs: 15
  max_epochs: 100
debug:
  accelerator: cpu
  limit_train_batches: 10
  limit_val_batches: 10
  log_every_n_steps: 1
  max_epochs: 2
experiment:
  accelerator: gpu
  strategy: ddp_find_unused_parameters_true
callbacks:
  - class_path: pytorch_lightning.callbacks.EarlyStopping
    init_args:
      patience: 5
      monitor: val_energy
      mode: min
      verbose: True
      check_finite: False
  - class_path: pytorch_lightning.callbacks.ModelCheckpoint
    init_args:
      monitor: val_energy
      save_top_k: 3
  - class_path: matsciml.lightning.callbacks.GradientCheckCallback
  - class_path: matsciml.lightning.callbacks.SAM
loggers:
  - class_path: pytorch_lightning.loggers.CSVLogger # can omit init_args['save_dir'] for auto directory
```



## Dataset Config
Similar to the trainer config, the dataset config has sections for debug and full experiments. Dataset paths, batch size, num workers, seed, and other relevant arguments may be set here. The available target keys for training are included. Other settings such as `normalization_kwargs` and `task_loss_scaling` may be set here under the `task_args` tag.
```yaml
dataset: CMDataset
debug:
  batch_size: 4
  num_workers: 0
experiment:
  test_split: ''
  train_path: ''
  val_split: ''
target_keys:
- energy
- symmetry_number
- symmetry_symbol
task_args:
  normalize_kwargs:
    energy_mean: 1
    energy_std: 1
  task_loss_scaling:
    energy: 1
```

## Model Config
Models available in matsciml my be DGL, PyG, or pointcloud based. Each model it named with its supported backend, as models may have more than one variety. In some instances, similar to the `trainer.yaml` config, a `class_path` and `init_args` need to be specified. Additionally, modules may need to be specified without initialization which may be done by using the `class_instance` tag. Finally, all transforms that a model should use should be included in the model config.
```yaml
encoder_class:
  class_path: matsciml.models.TensorNet
encoder_kwargs:
  element_types:
    class_path: matsciml.datasets.utils.element_types
  num_rbf: 32
  max_n: 3
  max_l: 3
output_kwargs:
  lazy: False
  input_dim: 64
  hidden_dim: 64
transforms:
  - class_path: matsciml.datasets.transforms.PeriodicPropertiesTransform
    init_args:
      cutoff_radius: 6.5
      adaptive_cutoff: True
  - class_path: matsciml.datasets.transforms.PointCloudToGraphTransform
    init_args:
      backend: dgl
      cutoff_dist: 20.0
      node_keys:
        - "pos"
        - "atomic_numbers"
```

## CLI Parameter Updates
Certain parameters may be updated using the cli. The `-c, --cli_args` argument may be used, and the parameter must be specified as `[config].parameter.value`. The config may be `trainer`, `model`, or `dataset`. For example, to update the batch size for a debug run:

`python training_script.py --debug --cli_args dataset.debug.batch_size.16`

 Only arguments which contain dict: [str, int, float] mapping all the way through to the target value may be updated. Parameters which map to lists at any point are not updatable through `cli_args`, for example callbacks, loggers, and transforms.
