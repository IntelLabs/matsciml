from __future__ import annotations

from os import PathLike
from pathlib import Path
from typing import Any
from json import dumps

import click

from matsciml.datasets.base import BaseLMDBDataset


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


if __name__ == "__main__":
    main()
