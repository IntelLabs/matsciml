
# Dataset implementation in Open MatSciML Toolkit

This document serves as a short overview of the dataset portion of the Open MatSciML Toolkit pipeline is designed,
both for the sake of maintenance as well as to serve as a reference for subsequent dataset implementations.

## Design philosophy

Here is a quick check overview of key elements of designing a dataset in the Open MatSciML Toolkit. For
more detailed descriptions, please read the later sections!

- Inheritance/object-oriented programming is used to maximize code reuse.
- Dataset methods should operate on *single* data samples, and are expected to return a `DataDict`, which is simply a dictionary consisting of string keys and various data formats (e.g. `float`, `torch.Tensor`).
- If possible, return a point cloud sample as it does not require specific graph backends like PyG or DGL. Transforms are written to convert between representations.
- Implement the *bare minimum* logic for `data_from_key` and use *transforms* to achieve modular behavior (e.g. convert point clouds to graphs, add optional properties).
- When possible, use [consistent keys for properties](#common-key-names-for-data-properties).
- Nest training targets under the `targets` key of your `DataDict`. This makes it explicit which tensors are intended to be used as outputs rather than input data, both for the pipeline as well as other users and maintainers.
  - The `target_keys` property categorizes targets into continuous (`regression`) and binary (`classification`) keys, which are subsequently used by the task abstractions.
- Commit a "development set" amount of data samples to the repository. This is typically around ~200 samples, and provides a straightforward way to test various stages of the pipeline, develop models offline, etc.
- Develop unit tests as you go, testing individual dataset functionality as you develop it.

## Integration checklist

The following is a checklist to help guide you through the implementation of a new dataset.

- [ ] Created submodule folder structure within `matsciml.datasets`
- [ ] Converted existing data to LMDB format
- [ ] Created minimal devset
- [ ] Added devset path to `MANIFEST.in`
- [ ] Created new dataset interface by subclassing `matsciml.datasets.base.BaseLMDBDataset`
- [ ] Implemented unit tests for new dataset class
- [ ] Test in minimal training loop
- [ ] Add your dataset import to `matsciml/datasets/__init__.py`

If you think your implementation will benefit the community (hint: almost definitely), we encourage
you to add the following steps as well:

- [ ] A dataset description added [`DATASETS.md]`](./DATASETS.md), briefly describing the dataset, details regarding its licensing, how and where it was sourced, and any implementation details you deem important
- [ ] Make sure the dataset is publicly available: Zenodo, Harvard Dataverse, etc. are repositories with version control and DOI generation
  - You can choose to upload the raw data, or in LMDB format. The former allows others to use the dataset without needing Open MatSciML Toolkit, but we'll ask you to make sure a preprocessing script is available to convert the data into LMDB after retrieval.

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

### Implementing dataset logic

With your data contained in an LMDB file, the real meat of your implementation now will be to
subclass `matsciml.datasets.base.BaseLMDBDataset` and create your own `NewMaterialsDataset`,
overriding abstract methods with concrete ones. The point of this is that a lot of common
functions are implemented just once in `BaseLMDBDataset`, and you don't need to do so again
for subsequent implementations.

`BaseLMDBDataset` in turn is actually just a subclass of native PyTorch's `Dataset` class;
if you are familiar with using `Dataset`, the only abstract methods that need to be
overridden are `__len__` and `__getitem__`, both of which are already implemented by
`BaseLMDBDataset`. The former will check across all your LMDB files and provide a total
number of data samples, while the latter does three things:

1. Check if `metadata` exists in LMDB files, and within it, whether a `preprocessed` entry is set to `True`.
2. If the dataset is preprocessed, read from the LMDB file directly; otherwise, execute `data_from_key` to retrieve a data sample from one of the LMDB files.
3. Perform any and all transformations on the retrieved data sample.

With that said, at the bare minimum your new dataset *may* require you to implement `data_from_key`,
if your dataset requires additional logic, such as reshaping tensors, type casting, and so on.
In that case, the most straightforward way to do so is just to re-use the existing method, and
append whatever extra things you need to do afterwards assuming you are subclassing `BaseLMDBDataset`:

```python
from matsciml.common.types import DataDict
from matsciml.common.registry import registry
from matsciml.datasets import utils

@registry.register_dataset("NewMaterialsDataset")
class NewMaterialsDataset(BaseLMDBDataset):
    def data_from_key(self, lmdb_index: int, sub_index: int) -> DataDict:
        data = super().data_from_key(lmdb_index, sub_index)
        # check to make sure 3D coordinates
        assert data["pos"].size(-1) == 3
        # typecast atomic numbers
        atom_numbers = torch.LongTensor(data["atomic_numbers"])
        # convert to bespoke +/- combination of one-hot encodings
        pc_features = utils.point_cloud_featurization(
            atom_numbers[src_nodes], atom_numbers[dst_nodes], 100
        )
        data["pc_features"] = pc_features
        return data
```

In this example, the original `data_from_key` is good enough: under the hood, the `super` method
ultimately calls `utils.get_data_from_index`, which just unpickles data contained at a particular
`index` value - in other words, whatever you saved in the LMDB conversion under `index` just gets
regurgitated. We then perform some checks and conversions on the tensors as needed. The `registry`
is also used here to decorate the class, which allows the rest of Open MatSciML Toolkit to recall
the class simply from a string (e.g. in the Lightning modules).

#### A note on data packing

We are looking at standardizing a data sample, but in the mean time the `DataDict` annotation
roughly describes the expected structure; a nested dictionary. The general packing strategy
(subject to improvement) is the following:

- For graph structures, pack node/edge attributes into a `DGLGraph` or `PyGGraph` structure, which ensures they are batched correctly.
  - Graph level attributes and features can be left at the top level of `DataDict`
- For point cloud structures, update the `collate_fn` method to include additional `pad_keys` to denote which key/tensors need to be padded for batching. See `LiPSDataset` as an example of this.
- See [this table](#common-key-names-for-data-properties) to standardize naming; e.g. atomic coordinates as `pos`
- Pack tensors intended as labels inside a `targets` dictionary
- For every key packed into `targets`, add them to the `target_keys` class *property* under the appropriate category. These classifications are used by the training pipeline for metric evaluation.

Below is an example of what is returned from `MaterialsProjectDataset.from_devset()`,
a point cloud representation used to illustrate what the structure can look like:

```python
>>> from matsciml.datasets import MaterialsProjectDataset
>>> dset = MaterialsProjectDataset.from_devset()
>>> sample = dset.__getitem__(0)
>>> for key, value in sample.items():
...     print(key, type(value))
...
pos <class 'torch.Tensor'>
atomic_numbers <class 'torch.Tensor'>
pc_features <class 'torch.Tensor'>
sizes <class 'int'>
src_nodes <class 'torch.Tensor'>
dst_nodes <class 'torch.Tensor'>
distance_matrix <class 'torch.Tensor'>
natoms <class 'int'>
lattice_features <class 'dict'>
targets <class 'dict'>
target_types <class 'dict'>
dataset <class 'str'>
```

Here, `pos`, `atomic_numbers` are standard features of a material structure. Additional
properties that might help downstream development include `natoms` (so it's unambiguous),
and `distance_matrix`. `targets` is a dictionary of key/value pairs:

```python
sample["targets"] = {"band_gap": 2.1}
```

And `target_keys` is nested dictionary:

```python
{"regression": ["band_gap"]}
```

Note, you should not write `target_keys` inside the `data_from_key`, and instead update
the class property:

```python
class NewMaterialsDataset:
    @property
    def target_keys(self) -> dict[str, list[str]]:
        # can also add a classification dict, holding binary label keys
        return {"regression": ["energy", "band_gap", "other_continuous_label"]}
```

You might also be wondering about what the `dataset` key refers to: this is automatically
appended by the `BaseLMDBDataset` method, which stores the classname of the dataset.
This is used during multidataset training to differentiate between samples, and correctly
map tasks with datasets.

#### Preprocessing

As mentioned earlier, there is some support for adding `preprocessed=True` as metadata. The idea
behind this is to provide some rudimentary way to do computationally expensive transforms to
your dataset, and cache the result. There is currently no standardized way of doing so, but
the recommended approach would be to loop over your `NewMaterialsDataset` with a simple `for`
loop, write out to an LMDB file as shown [in the earlier section](#creating-lmdb-files), and
writing an additional `metadata` key with a value of `{preprocessed: True}`. When you use `NewMaterialsDataset`
and point to the preprocessed LMDB file, it should bypass all of the `data_from_key` logic, allowing you
to skip intensive portions of data retrieval.

### Adding a development set

We recommend committing a minimal amount of data to the repository as a way to streamline
debugging and usage across the full Open MatSciML Toolkit pipeline. We won't go into specific
details on *how* to generate the devset, as it can be as simple as writing only the first
200 samples as shown in the [LMDB conversion section](#creating-lmdb-files). You can then move the resulting
folder structure to the `devset` [contained in your submodule](#implementing-a-new-dataset),
and update the `MANIFEST.in` file contained in the repository root folder (where `pyproject.toml` is),
which ensures that the `devset` folder will be included in `pip` installs.

`BaseLMDBDataset` implements a class method called `from_devset`, which relies on a private
attribute for your dataset called `__devset__`. This is basically a hardcoded path to
your `devset` path; you can see how other datasets implement this, but assuming you have
the correct folder structure set up, you will just need to add the following to your
class definition:

```python
from pathlib import Path

from matsciml.common.registry import registry
from matsciml.datasets.base import BaseLMDBDataset

@registry.register_dataset("NewMaterialsDataset")
class NewMaterialsDataset(BaseLMDBDataset):
    __devset__ = Path(__file__).parents[0].joinpath("devset")

    ...
```

As a brief explanation, `__file__` points to the module file, and this pattern is simply
retrieving the relative path to the `devset` folder assuming it's at the same level
as the module file. With that, you'll be able to retrieve your development set from anywhere
with `NewMaterialsDataset.from_devset()`.

### Writing unit tests

Unit testing is a pretty expansive area, and difficult to standardize and convey *what* and *how*
to test well. As a foundational basis, the tests should aim to make sure core functions work
consistently, and check for things that may commonly go wrong such as tensor shapes, missing
keys, and so on. We encourage you to look at how other datasets implement unit tests with `pytest`,
and see if you can adopt elements from them.

While it's not strictly perfect, we encourage you to use `devset`s for your unit testing. This
will allow you to write tests as you develop the dataset interface in an iterative way; as
a skeleton example, a minimal test suite could look like this:

```python
from matsciml.datasets.new_materials_data import NewMaterialsDataset


def test_devset_init():
    """Test whether or not the dataset can be created from devset"""
    dset = NewMaterialsDataset.from_devset()


def test_devset_read():
    """Ensure we can read every entry in the devset"""
    dset = NewMaterialsDataset.from_devset()
    num_samples = len(dset)
    for index in range(num_samples):
        sample = dset.__getitem__(index)


def test_devset_keys():
    """Ensure the devset contains keys and structure we expect"""
    dset = NewMaterialsDataset.from_devset()
    sample = dset.__getitem__(50)
    for key in ["pos", "atomic_numbers", "force"]:
        assert key in sample
    # we know this dataset has regression data
    assert "regression" in sample["targets"]
    for key in ["bandgap", "fermi_energy"]:
        assert key in sample["targets"]["regression"]
```

You can also encourage re-use with things like `pytest.fixture`s, but things can get pretty
complicated quickly, so for pull requests, the bar is likely not going to be too high for
these dataset unit tests. At a minimum, we ask you to replicate the tests above for your
dataset, but highly encourage more expansive testing (you can never have too much!).

### Final integration

The last part of integrating your new dataset is using it to train a model! From this
point onwards it depends entirely on what you want to do with your data, and so we only
provide a very minimal example.

```python
import pytorch_lightning as pl

from matsciml.datasets import new_materials_data
from matsciml.lightning.data_utils import MatSciMLDataModule

# this uses the devset for training
data_module = MatSciMLDataModule.from_devset(
    dataset="NewMaterialsDataset",
    batch_size=8
)
# configure your task
task = ...
# fast_dev_run performs a minimal training loop with 10 steps
# without checkpointing, logging, etc.
trainer = pl.Trainer(fast_dev_run=10)
trainer.fit(task, datamodule=data_module)
```

With this, you should be able to test the end-to-end functionality of your dataset;
here's hoping everything works!

## Documenting new datasets

When contributing a new dataset, please add the following details to [`DATASETS.md`](./DATASETS.md):

- General description: its nominal purpose, how the data was generated, and its distribution license (e.g. MIT)
- How to download the data from a public repository, and if applicable, how to preprocess it locally
- Relevant citations for the dataset: arXiv, DOIs, paper references
- Data keys and descriptions of each key/value mapping. If appropriate, *please include physical units*.

## Common key names for data properties

This section is kept to try and encourage consistent key naming between datasets.

The intention is to expand and update this as more datasets are integrated.

| Name | Description |
|---|---|
| `atomic_numbers` | Atomic numbers of nodes/points |
| `pos` | Cartesian coordinates of nodes/points |
| `force` | Force labels for nodes/points |
| `pc_features` | Per-atom features for point clouds |

[lmdb]: https://lmdb.readthedocs.io/en/release/
