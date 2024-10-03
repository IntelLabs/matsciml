from __future__ import annotations

import code
from os import PathLike
from pathlib import Path
from typing import Any, Literal
from json import dumps

import click

from matsciml import datasets  # noqa: F401
from matsciml.common.registry import registry
from matsciml.datasets.base import BaseLMDBDataset
from matsciml.datasets.transforms import (
    PeriodicPropertiesTransform,
    PointCloudToGraphTransform,
)


__available_datasets__ = sorted(list(registry.__entries__["datasets"].keys()))


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
    dataset = target_class(Path(lmdb_dir).resolve(), transforms=transforms)  # noqa: F401
    code.interact(local=locals())


if __name__ == "__main__":
    main()
