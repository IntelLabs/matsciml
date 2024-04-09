from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
from functools import partial
from typing import Callable
from math import ceil
from os import makedirs
from importlib.util import find_spec
from logging import getLogger

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from pymatgen.core import Composition

logger = getLogger(__file__)


def get_atom_set(formula: str) -> list[str]:
    """
    Convience function that returns unique atoms within a composition.

    Parameters
    ----------
    formula : str
        Reduced formula to be passed into a ``Composition``

    Returns
    -------
    list[str]
        Unsorted list of atom symbols
    """
    comp = Composition(formula)
    return list(comp.get_el_amt_dict().keys())


def has_atom_in_set(reduced_formula: str, atom_set: set[str]) -> bool:
    """
    Given a set of atoms to check against, make sure that at least
    one is contained in a reduced formula.

    The idea with this function is to find atoms that will be
    missing from the training split.

    Parameters
    ----------
    reduced_formula : str
        Reduce formula to check for atoms contained in ``atom_set``
    atom_set : set[str]
        Set of atoms to check against

    Returns
    -------
    bool
        True if any atoms in ``atom_set`` are present in ``reduced_formula``
    """
    return any([atom in reduced_formula for atom in atom_set])


def add_missing_atoms_to_train(
    corrected_df: pd.DataFrame,
    check_func: Callable,
    rng: np.random.Generator,
    train_fraction: float = 0.7,
) -> None:
    """
    Look for atoms that are missing in the train set and include them.

    It's unclear whether it makes sense to validate/test on structures
    with unseen atoms, and so this function will ensure that the training
    set includes all atom types within a dataset, even if it means
    moving compositions from validation and testing. Function modifies
    the input table inplace.

    This works by applying a ``check_func`` across the reduced compositions,
    and identifying rows that include atom types not present in training.

    In the event that only a single composition exists, we move those
    out of val/test and into training. Otherwise, we take ``train_fraction``
    and essentially perform a nested split, attempting to move this target
    fraction of samples into training.

    Parameters
    ----------
    corrected_df : pd.DataFrame
        Dataframe that has already gone through the first round of splits.
    check_func : Callable
        Function that determines whether the reduced composition exists
        in the training set.
    rng : np.random.Generator
        Instances of a NumPy random number generator, used to shuffle
        indices where multiple compositions are identified to further
        split.
    train_fraction : float
        Fraction of samples to move into training, should there be
        multiple compositions.
    """
    # find rows that don't have atoms in Set
    mask = corrected_df["reduced_composition"].apply(check_func)
    missing_rows = corrected_df[mask]
    unique_comps = missing_rows["composition_index"].unique()
    # in the simple case, we just have one composition to worry about
    # and just put it in training split
    if len(unique_comps) == 1:
        corrected_df.loc[mask, "split"] = 0
    else:
        rng.shuffle(unique_comps)
        num_comps = len(unique_comps)
        train_indices = np.arange(num_comps)[: ceil(train_fraction * num_comps)]
        train_comps = unique_comps[train_indices]
        for index in train_comps:
            corrected_df.loc[corrected_df["composition_index"] == index, "split"] = 0
    return None


if find_spec("seaborn"):
    import seaborn as sns

    def make_ecdf_plots(
        dataframe: pd.DataFrame, ignore_keys: list[str] = []
    ) -> plt.Figure:
        """
        Generate marginalized empirical cumulative distribution plots for inspection.

        These plots are useful for revealing underlying data distributions;
        for example, whether a property is normally, uniformly, or exponentially
        distributed. This can be used to guide label normalization practiecs.

        Parameters
        ----------
        dataframe : pd.DataFrame
            Pandas dataframe to inspect.
        ignore_keys : list[str]
            A list of keys that exist in ``dataframe`` to exclude from plotting.

        Returns
        -------
        plt.Figure
            Matplotlib ``Figure`` object
        """
        target_keys = list(filter(lambda x: x not in ignore_keys, dataframe.keys()))
        num_rows = len(target_keys) // 3
        # make figure with 3 columns and variable rows
        fig, axarray = plt.subplots(num_rows, 3, figsize=(10, 10), sharey=True)

        for key, ax in zip(target_keys, axarray.flatten()):
            sns.ecdfplot(dataframe, x=key, ax=ax, hue="split")
        return fig


def subtract_reference_energy(dataframe: pd.DataFrame, key: str) -> None:
    """
    Uses ``groupby`` to find shared compositions to compute reference
    energies and globally (within a composition) subtract it.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Target dataframe to operate on.
    key : str
        Label key to compute the reference energy, and subtract from.
    """
    assert key in dataframe.keys(), f"{key} specified was not in dataframe!"
    # find the minimum energy for a given composition
    dataframe["reference_energy"] = dataframe.groupby("composition_index")[
        key
    ].transform("min")
    dataframe["reference_structure"] = dataframe.groupby("composition_index")[
        key
    ].transform("idxmin")
    dataframe["relative_energy"] = dataframe[key] - dataframe["reference_energy"]
    return None


def main(
    input_path: Path,
    seed: int,
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
    key: str | None,
):
    rng = np.random.default_rng(seed)
    assert (
        sum([train_fraction, val_fraction, test_fraction]) == 1.0
    ), f"Split fractions do not add up! Train: {train_fraction}, Test: {test_fraction}, Val: {val_fraction}"
    df = pd.read_csv(input_path)
    unique_compositions = df["composition_index"].unique()
    # shuffle and split
    rng.shuffle(unique_compositions)
    num_comps = len(unique_compositions)
    # calculate the indices at which we will split the compositions
    split_at = np.cumsum(
        np.array([train_fraction, val_fraction, test_fraction]) * num_comps
    )
    train_comps, val_comps, test_comps, _ = np.array_split(
        unique_compositions, split_at.astype(int)
    )
    # set categorical indices to each composition
    all_indices = np.zeros_like(unique_compositions)
    for split, value in zip([train_comps, val_comps, test_comps], [0, 1, 2]):
        all_indices[split] = value
    split_mapping = {
        comp_index: split_index for comp_index, split_index in enumerate(all_indices)
    }
    # remap all the split categories to the full dataset
    df["split"] = df["composition_index"].map(split_mapping)
    # sodium nitride is parsed by pandas as not a number
    df["reduced_composition"].replace(np.nan, "NaN", inplace=True)
    # now we have to determine which atom types are missing from train set
    df["atom_sets"] = df["reduced_composition"].apply(get_atom_set)
    # this aggregates unique atoms within splits
    grouped_atom_sets = df.groupby("split")["atom_sets"].agg(sum).values
    uncorr_train_atom_set, uncorr_val_atom_set, uncorr_test_atom_set = [
        set(group) for group in grouped_atom_sets
    ]
    missing_val_atoms = uncorr_val_atom_set.difference(uncorr_train_atom_set)
    missing_test_atoms = uncorr_test_atom_set.difference(uncorr_train_atom_set)
    # corresponds to the full atom set, not used right now
    _ = uncorr_train_atom_set | uncorr_val_atom_set | uncorr_test_atom_set
    val_check = partial(has_atom_in_set, atom_set=missing_val_atoms)
    test_check = partial(has_atom_in_set, atom_set=missing_test_atoms)
    atom_corrected_df = df.copy()
    add_missing_atoms_to_train(atom_corrected_df, val_check, rng, train_fraction)
    add_missing_atoms_to_train(atom_corrected_df, test_check, rng, train_fraction)
    # if a key was specified, try and subtract the reference energy
    if key:
        subtract_reference_energy(atom_corrected_df, key)
    # get the dataset name and use it as the directory
    output_dir = Path(input_path.name.split("_")[0])
    makedirs(output_dir, exist_ok=True)
    atom_corrected_df.to_csv(str(output_dir.joinpath("statistics_and_splits.csv")))
    # see if seaborn is available for plotting with
    if find_spec("seaborn"):
        fig = make_ecdf_plots(
            atom_corrected_df, ignore_keys=["index", "reduced_composition"]
        )
        fig.tight_layout()
        fig.savefig(
            str(output_dir.joinpath("ecdf_plots.png")), dpi=300, transparent=True
        )
    else:
        logger.warning("seaborn installation not found; no ECDF plots produced.")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("input_path", type=Path, help="Input summary CSV file.")
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=31636,
        help="Random seed to use for shuffling.",
    )
    parser.add_argument(
        "-t",
        "--train_fraction",
        type=float,
        default=0.75,
        help="Approximate training split in compositions.",
    )
    parser.add_argument(
        "-v",
        "--val_fraction",
        type=float,
        default=0.15,
        help="Approximate validation split in compositions.",
    )
    parser.add_argument(
        "-e",
        "--test_fraction",
        type=float,
        default=0.10,
        help="Approximate testing split in compositions.",
    )
    parser.add_argument(
        "-k",
        "--key",
        type=str,
        default=None,
        help="Key to use for reference energy calculation.",
    )

    args = parser.parse_args()
    main(
        args.input_path,
        args.seed,
        args.train_fraction,
        args.val_fraction,
        args.test_fraction,
        args.key,
    )
