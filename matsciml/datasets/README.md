
# Dataset implementation in Open MatSciML Toolkit

This document serves as a short overview of the dataset portion of the Open MatSciML Toolkit pipeline is designed,
both for the sake of maintenance as well as to serve as a reference for subsequent dataset implementations.

## Design philosophy

- Inheritance/object-oriented programming is used to maximize code reuse.
- Dataset methods should operate on *single* data samples, and are expected to return a `DataDict`, which is simply a dictionary consisting of string keys and various data formats (e.g. `float`, `torch.Tensor`).
- If possible, return a point cloud sample as it does not require specific graph backends like PyG or DGL. Transforms are written to convert between representations.
- Implement the *bare minimum* logic for `data_from_key` and use *transforms* to achieve modular behavior (e.g. convert point clouds to graphs, add optional properties).
- When possible, use [consistent keys for properties](#common-key-names-for-data-properties).
- Nest training targets under the `targets` key of your `DataDict`. This makes it explicit which tensors are intended to be used as outputs rather than input data, both for the pipeline as well as other users and maintainers.
  - The `target_keys` property categorizes targets into continuous (`regression`) and binary (`classification`) keys, which are subsequently used by the task abstractions.
- Commit a "development set" amount of data samples to the repository. This is typically around ~200 samples, and provides a straightforward way to test various stages of the pipeline, develop models offline, etc.
- Develop unit tests as you go, testing individual dataset functionality as you develop it.

## Implementing a new dataset

The first step in contributing a new dataset is reading the [contribution guide](../../CONTRIBUTING.md) to configure your environment and such. At a high level,
fork this repository, clone your fork locally, then create a new branch for code implementations. Assuming you have done this, the first
step will be to create a submodule for your new dataset in `matsciml.datasets`; in the `/matsciml/datasets` folder, create a new directory:

```console
./matsciml
├── common
│   └── relaxation
├── datasets
│   ├── carolina_db
│   ├── embeddings
│   ├── lips
│   ├── materials_project
│   ├── nomad
│   ├── oqmd
│   ├── symmetry
│   ├── tests
│   ├── transforms
│   └── <new_dataset>
```

In your `<new_dataset>` folder, we'll need the following files:

```console
./matsciml
├── datasets
    ├── <new_dataset>
        ├── __init__.py
        ├── dataset.py
        ├── tests
        └── devset
```

This includes a `tests` folder for your `pytest` unit tests, and a `devset`
folder which will contain a small `.lmdb` file you'll create later holding
your development dataset. If your dataset involves querying a remote API,
we recommend creating a separate `api.py` module; see our Materials Project
or NOMAD implementations for examples there.

### Creating LMDB files

Chances are, your dataset is not contained within LMDB files. The main advantages
of LMDB are performance (as a binary format) and scalability (both in size and
distributed settings). While a good understanding of how LMDB works is (hopefully)
not necessary, we recommend perusing their [documentation][lmdb].

To convert your data into LMDB files, we've provided some functions for reading
and writing to LMDB - as part of developing the dataset you'll need to write a
small script to convert the original data format into an LMDB file. As a very
simple illustrative example, if we had a dataset stored as JSON structures
contained in a single file, the script might look like this:

```python
from json import load
# utils module contains LMDB routines
from matsciml.datasets import utils

# assume json_data is a list of dicts
with open("dataset.json", "r"") as read_file:
  json_data = load(read_file)

# set a target directory that will hold all the LMDB files
lmdb_target_dir = "<path_to_datasets>/matsciml/new_materials_data/train"

utils.parallel_lmdb_write(lmdb_target_dir, json_data, num_procs=8)
```

The last line will divide `json_data` amongst `num_procs` workers, each creating an LMDB
file in `lmdb_target_dir`; in this case, we're working on the training split and will
create `data.0000-7.lmdb` inside the `train` folder.

If this high level functionality doesn't work, you can manually implement a loop
over samples and write the data out like so:

```python
lmdb_file = utils.connect_lmdb_write(lmdb_target_dir + "/data.lmdb")

for index, data in enumerate(json_data):
  utils.write_lmdb_data(index, data, lmdb_file)
```

This uses `index` (as with other datasets) as the key to a particular data sample. You can use
arbitrary key names, such as `metadata` to include extra notes on the dataset as well (e.g.
when and how it was accessed, etc.), however *only integers will automatically recognized as
data samples* by methods like `utils.get_lmdb_data_keys` and by extension, `utils.get_lmdb_data_length`
which is used by the `BaseLMDBDataset` class to determine the number of samples via `__len__`.


## Inheritance

## Common key names for data properties

| Name | Description |
|---|---|
| `atomic_numbers` | Atomic numbers of nodes/points |
| `pos` | Cartesian coordinates of nodes/points |
| `force` | Force labels for nodes/points |
