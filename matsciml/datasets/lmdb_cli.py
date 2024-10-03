from __future__ import annotations

import code
from os import PathLike
from pathlib import Path
from typing import Any, Literal
from json import dumps, dump
from collections import deque
from logging import getLogger

import click
import torch
import numpy as np
from tqdm import tqdm
from torch_geometric.data import Data as PyGGraph
from torch_geometric.utils import degree

from matsciml import datasets  # noqa: F401
from matsciml.common.registry import registry
from matsciml.datasets.base import BaseLMDBDataset
from matsciml.datasets.transforms import (
    PeriodicPropertiesTransform,
    PointCloudToGraphTransform,
)


__available_datasets__ = sorted(list(registry.__entries__["datasets"].keys()))


class Accumulator:
    def __init__(self, name: str, maxlen: int):
        self._queue = deque(maxlen=maxlen)
        self.name = name
        self._logger = getLogger(f"lmdb_cli.Accumulator.{name}")

    def append(self, data: Any):
        self._queue.append(data)

    @property
    def array(self) -> np.ndarray:
        data = list(self._queue)
        try:
            if isinstance(data[0], torch.Tensor):
                data = torch.vstack(data).numpy()
            elif isinstance(data[0], np.ndarray):
                data = np.vstack(data)
        except Exception:
            data = np.concatenate([sample.flatten() for sample in data])
        data = np.array(data)
        return data

    def __str__(self) -> str:
        data = self.array
        mean = data.mean()
        std = data.std()
        return f"{mean:.3f}±{std:.3f}"

    def __repr__(self) -> str:
        return dumps(self.to_json())

    def to_json(self) -> dict[str, str | float]:
        data = self.array
        mean = data.mean()
        std = data.std()
        return {"name": self.name, "mean": float(mean), "std": float(std)}


def _update_accumulators(
    sample: dict[str, Any], accumulators: dict[str, Accumulator], maxlen: int = 10
):
    """
    Helper function that will recursively update (or create) accumulators
    for numeric data.

    Parameters
    ----------
    sample : dict[str, Any]
        Data sample from a dataset.
    accumulators : dict[str, Accumulator]
        Dictionary mapping of accumulators. Can be empty, as
        keys that are detected that do not exist already will
        be created with ``maxlen`` as the running average.
    maxlen : int
        Length of the buffer used to compute the running average.
    """
    for key, value in sample.items():
        if isinstance(value, dict):
            _update_accumulators(value, accumulators)
        elif isinstance(value, (int, float, np.ndarray, torch.Tensor)):
            if key not in accumulators:
                accumulators[key] = Accumulator(key, maxlen=maxlen)
            a = accumulators[key]
            a.append(value)
        else:
            pass
    # for pyg so far, we also accumulate graph node/edge metrics
    if "graph" in sample:
        graph = sample["graph"]
        if isinstance(graph, PyGGraph):
            agg_dict = {
                "num_nodes": graph.num_nodes,
                "avg_degree": degree(graph.edge_index[:, 0], graph.num_nodes)
                .float()
                .mean(),
            }
            for key, value in agg_dict.items():
                if key not in accumulators:
                    accumulators[key] = Accumulator(key, maxlen=maxlen)
                a = accumulators[key]
                a.append(value)


def _make_dataset(
    lmdb_dir: PathLike,
    dataset_type: str | None,
    periodic: bool,
    radius: float,
    adaptive_cutoff: bool,
    graph_backend: Literal["pyg", "dgl"] | None,
):
    """
    Abstracted out function that will instantiate a dataset object.

    Parameters
    ----------
    lmdb_dir : PathLike
        Path to an LMDB folder structure.
    dataset_type : str, optional
        Class name for the dataset to interpret the LMDB data. By
        default is ``None``, which uses ``BaseLMDBDataset`` to
        load the data. Checks against the ``matsciml`` registry for
        available datasets.
    periodic : bool, default True
        Whether to enable periodic properties transform.
    radius : float
        Cut-off radius used by the periodic property transform.
    adaptive_cutoff : bool, default True
        Whether to enable the adapative cut-off in the periodic
        properties transform.
    graph_backend : Optional, Literal['pyg', 'dgl']
        Optional choice for graph backend to use. The default is ``pyg``,
        which emits PyTorch Geometric graphs.
    """
    transforms = []
    if periodic:
        transforms.append(PeriodicPropertiesTransform(radius, adaptive_cutoff))
    if graph_backend:
        transforms.append(PointCloudToGraphTransform(graph_backend))
    target_class = (
        BaseLMDBDataset
        if not dataset_type
        else registry.get_dataset_class(dataset_type)
    )
    dataset = target_class(Path(lmdb_dir).resolve(), transforms=transforms)
    return dataset


def _recurse_dictionary_types(input_dict: dict[Any, Any]) -> dict[Any, str]:
    return_dict = {}
    for key, value in input_dict.items():
        if isinstance(value, dict):
            return_dict[key] = _recurse_dictionary_types(value)
        else:
            return_dict[key] = type(value).__name__
    return return_dict


@click.group()
def main() -> None:
    """
    A command-line interface for inspecting Open MatSciML Toolkit LMDB files.

    Subcommands provide a number of utilities that can be helpful for checking
    the contents of an LMDB dataset in the command line without necessarily
    writing bespoke code.
    """
    ...


@main.command()
@click.argument(
    "lmdb_dir", type=click.Path(exists=True, file_okay=False, dir_okay=True)
)
def print_structure(lmdb_dir: PathLike):
    """
    Retrieve the first sample in an LMDB set and print out its keys.

    This is useful for checking whether or not the key you expect
    is present in the data. This function uses ``BaseLMDBDataset``
    to load in the data, without assuming any particular dataset type.

    Parameters
    ----------
    lmdb_dir : PathLike
        String or Path object pointing to an LMDB directory.
    """
    dset = BaseLMDBDataset(Path(lmdb_dir).resolve())
    sample = dset.__getitem__(0)
    sample_struct = _recurse_dictionary_types(sample)
    click.echo("Data in LMDB sample corresponds to the following structure:")
    click.echo(dumps(sample_struct, sort_keys=True, indent=2))


@main.command()
@click.argument(
    "lmdb_dir", type=click.Path(exists=True, file_okay=False, dir_okay=True)
)
@click.argument("index", type=int)
@click.option(
    "-d",
    "--dataset_type",
    type=click.Choice(__available_datasets__),
    default=None,
    help="Dataset class name to use to map the data.",
)
@click.option(
    "-p",
    "--periodic",
    is_flag=True,
    default=True,
    help="Flag to disable the periodic transform.",
)
@click.option(
    "-r",
    "--radius",
    type=float,
    default=6.0,
    show_default=True,
    help="Cut-off radius for periodic property transform.",
)
@click.option(
    "-a",
    "--adaptive_cutoff",
    is_flag=True,
    default=True,
    help="Flag to disable the adaptive cutoff used in periodic transform.",
)
@click.option(
    "-g",
    "--graph_backend",
    type=click.Choice(["pyg", "dgl"], case_sensitive=False),
    default="pyg",
    help="Graph backend for transformation.",
)
def check_sample(
    lmdb_dir: PathLike,
    index: int,
    periodic,
    graph_backend,
    dataset_type,
    radius,
    adaptive_cutoff,
):
    """
    Given an LMDB path and a data index, perform optional transforms
    and print out what the specific sample contains.

    Parameters
    ----------
    lmdb_dir : PathLike
        Path to an LMDB folder structure.
    index : int
        Index to the data sample.
    dataset_type : str, optional
        Class name for the dataset to interpret the LMDB data. By
        default is ``None``, which uses ``BaseLMDBDataset`` to
        load the data. Checks against the ``matsciml`` registry for
        available datasets.
    periodic : bool, default True
        Whether to enable periodic properties transform.
    radius : float
        Cut-off radius used by the periodic property transform.
    adaptive_cutoff : bool, default True
        Whether to enable the adapative cut-off in the periodic
        properties transform.
    graph_backend : Optional, Literal['pyg', 'dgl']
        Optional choice for graph backend to use. The default is ``pyg``,
        which emits PyTorch Geometric graphs.
    """
    dataset = _make_dataset(
        lmdb_dir, dataset_type, periodic, radius, adaptive_cutoff, graph_backend
    )
    sample = dataset.__getitem__(index)
    for key, value in sample.items():
        click.echo(f"Key - {key}, value: {value}")


@main.command()
@click.argument(
    "lmdb_dir", type=click.Path(exists=True, file_okay=False, dir_okay=True)
)
@click.option(
    "-d",
    "--dataset_type",
    type=click.Choice(__available_datasets__),
    default=None,
    help="Dataset class name to use to map the data.",
)
@click.option(
    "-p",
    "--periodic",
    is_flag=True,
    default=True,
    help="Flag to disable the periodic transform.",
)
@click.option(
    "-r",
    "--radius",
    type=float,
    default=6.0,
    show_default=True,
    help="Cut-off radius for periodic property transform.",
)
@click.option(
    "-a",
    "--adaptive_cutoff",
    is_flag=True,
    default=True,
    help="Flag to disable the adaptive cutoff used in periodic transform.",
)
@click.option(
    "-g",
    "--graph_backend",
    type=click.Choice(["pyg", "dgl"], case_sensitive=False),
    default="pyg",
    help="Graph backend for transformation.",
)
def interactive(
    lmdb_dir: PathLike,
    dataset_type: str | None,
    periodic: bool,
    radius: float,
    adaptive_cutoff: bool,
    graph_backend: Literal["pyg", "dgl"] | None,
):
    """
    Loads an LMDB dataset and switches over to an interactive session.

    This allows you to quickly load up an LMDB dataset and perform some interactive
    debugging. The additional options provide control over graph creation (or lack
    thereof).

    You can subsequently iterate through the ``dataset`` variable however you
    wish, but typically with the ``__getitem__(<index>)`` method.

    Parameters
    ----------
    lmdb_dir : PathLike
        Path to an LMDB folder structure.
    dataset_type : str, optional
        Class name for the dataset to interpret the LMDB data. By
        default is ``None``, which uses ``BaseLMDBDataset`` to
        load the data. Checks against the ``matsciml`` registry for
        available datasets.
    periodic : bool, default True
        Whether to enable periodic properties transform.
    radius : float
        Cut-off radius used by the periodic property transform.
    adaptive_cutoff : bool, default True
        Whether to enable the adapative cut-off in the periodic
        properties transform.
    graph_backend : Optional, Literal['pyg', 'dgl']
        Optional choice for graph backend to use. The default is ``pyg``,
        which emits PyTorch Geometric graphs.
    """
    dataset = _make_dataset(
        lmdb_dir, dataset_type, periodic, radius, adaptive_cutoff, graph_backend
    )  #  noqa: F401
    code.interact(local=locals())


@main.command()
@click.argument(
    "lmdb_dir", type=click.Path(exists=True, file_okay=False, dir_okay=True)
)
@click.option(
    "-d",
    "--dataset_type",
    type=click.Choice(__available_datasets__),
    default=None,
    help="Dataset class name to use to map the data.",
)
@click.option(
    "-p",
    "--periodic",
    is_flag=True,
    default=True,
    help="Flag to disable the periodic transform.",
)
@click.option(
    "-r",
    "--radius",
    type=float,
    default=6.0,
    show_default=True,
    help="Cut-off radius for periodic property transform.",
)
@click.option(
    "-a",
    "--adaptive_cutoff",
    is_flag=True,
    default=True,
    help="Flag to disable the adaptive cutoff used in periodic transform.",
)
@click.option(
    "-g",
    "--graph_backend",
    type=click.Choice(["pyg", "dgl"], case_sensitive=False),
    default="pyg",
    help="Graph backend for transformation.",
)
@click.option(
    "-n",
    "--num_samples",
    type=int,
    default=None,
    help="If specified, corresponds to the maximum number of samples to compute with.",
)
@click.option(
    "-w",
    "--window_size",
    type=int,
    default=10,
    help="Window size for computing the running average over.",
)
def dump_statistics(
    lmdb_dir: PathLike,
    dataset_type: str | None,
    periodic: bool,
    radius: float,
    adaptive_cutoff: bool,
    graph_backend: Literal["pyg", "dgl"] | None,
    num_samples: int | None,
    window_size: int,
):
    """
    Loads an LMDB dataset and iterates through the dataset,
    computing a running average for numeric properties that
    updates interactively and written to a JSON file afterwards.

    The JSON file will be named after the dataset class used
    to interpret the data followed by the specific directory/split
    name and written to the current folder.

    Parameters
    ----------
    lmdb_dir : PathLike
        Path to an LMDB folder structure.
    dataset_type : str, optional
        Class name for the dataset to interpret the LMDB data. By
        default is ``None``, which uses ``BaseLMDBDataset`` to
        load the data. Checks against the ``matsciml`` registry for
        available datasets.
    periodic : bool, default True
        Whether to enable periodic properties transform.
    radius : float
        Cut-off radius used by the periodic property transform.
    adaptive_cutoff : bool, default True
        Whether to enable the adapative cut-off in the periodic
        properties transform.
    graph_backend : Optional, Literal['pyg', 'dgl']
        Optional choice for graph backend to use. The default is ``pyg``,
        which emits PyTorch Geometric graphs.
    num_samples : int, optional
        If provided, sets the maximum number of samples to iterate
        over.
    """
    dataset = _make_dataset(
        lmdb_dir, dataset_type, periodic, radius, adaptive_cutoff, graph_backend
    )
    accum = {}
    stats = []
    if not num_samples:
        num_samples = len(dataset)
    num_samples = min(num_samples, len(dataset))
    pbar = tqdm(range(num_samples), total=num_samples, unit=" samples")
    for index in pbar:
        sample = dataset.__getitem__(index)
        _update_accumulators(sample, accum, window_size)
        # now update the progress bar with the statistics
        desc = ""
        for key, a in accum.items():
            desc += f"{key}: {str(a)} "
        pbar.set_description(desc)
        stats.append([a.to_json() for a in accum.values()])
    with open(
        f"{dataset.__class__.__name__}-{Path(lmdb_dir).stem}_statistics.json", "w+"
    ) as write_file:
        dump(stats, write_file, indent=2)


if __name__ == "__main__":
    main()
