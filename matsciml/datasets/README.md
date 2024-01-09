
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
your development test.

## Inheritance

## Common key names for data properties

| Name | Description |
|---|---|
| `atomic_numbers` | Atomic numbers of nodes/points |
| `pos` | Cartesian coordinates of nodes/points |
| `force` | Force labels for nodes/points |
