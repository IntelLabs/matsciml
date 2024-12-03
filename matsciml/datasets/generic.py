from __future__ import annotations

from hashlib import blake2s
from functools import cache
from os import PathLike
from pathlib import Path
from typing import Any, Callable, Literal
from logging import getLogger

import h5py
from torch.utils.data import Dataset
from lightning import pytorch as pl

from matsciml.datasets.schema import DatasetSchema, DataSampleSchema


logger = getLogger("matsciml.datasets.MatSciMLDataset")


def write_data_to_hdf5_group(key: str, data: Any, h5_group: h5py.Group) -> None:
    """
    Writes data recursively to an HDF5 group.

    For dictionary data, we will create a new group and call this
    same function to write the subkey/values within the new group.

    For strings, the data is written to the ``attrs``.

    Parameters
    ----------
    key : str
        Key to write the data to; this will create a new
        ``h5py.Dataset`` object under this name within the group.
    data : Any
        Any data to write to the HDF5 group.
    h5_group : h5py.Group
        Instance of an ``h5py.Group`` to write datasets to.
    """
    if isinstance(data, dict):
        subgroup = h5_group.create_group(key)
        for subkey, subvalue in data.items():
            write_data_to_hdf5_group(subkey, subvalue, subgroup)
    elif isinstance(data, str):
        h5_group.attrs[key] = data
    else:
        h5_group[key] = data


class MatSciMLDataset(Dataset):
    def __init__(
        self,
        filepath: PathLike,
        transforms: list[Callable] | None = None,
        strict_checksum: bool = True,
    ):
        """
        Dataset class for generic ``MatSciMLDataset``s that use
        the data schema specifications.

        The main output of this class is mainly data loading from
        HDF5 files, parsing ``DatasetSchema`` metadata that are
        adjacent to HDF5 files, and returning data samples in the
        form of ``DataSampleSchema`` objects, which in principle
        should replace conventional ``DataDict`` (i.e. just plain
        dictionaries with arbitrary key/value pairs) that were used
        in earlier ``matsciml`` versions.

        Parameters
        ----------
        filepath : PathLike
            Filepath to a specific HDF5 file, containing a data split
            of either ``train``, ``test``, ``validation``, or ``predict``.
        transforms : list[Callable], optional
            If provided, should be a list of Python callable objects
            that will operate on the data.
        strict_checksum : bool, default True
            If ``True``, the dataset will refuse to run if it does not
            match any of the checksums contained in the metadata. This
            implementation does not **need** to know which split the data
            is, but has to match at least one of the specified splits.
            This can be disabled manually by setting to ``False``, but
            means the dataset can be modified.

        Raises
        ------
        RuntimeError:
            If no checksums in the metadata match the current data
            while ``strict_checksum`` is set to ``True``.
        """
        super().__init__()
        if not isinstance(filepath, Path):
            filepath = Path(filepath)
        self.filepath = filepath
        self.transforms = transforms
        if strict_checksum:
            metadata = self.metadata
            success = False
            for key, value in metadata.split_blake2s.items():
                if value == self.blake2s_checksum:
                    success = True
                    logger.debug(
                        f"Matched dataset checksum with {key} split from metadata."
                    )
            if not success:
                raise RuntimeError(
                    "Dataset checksum failed to validate against any splits in metadata."
                )

    @property
    def metadata(self) -> DatasetSchema:
        """
        Underlying ``DatasetSchema`` that should accompany all ``matsciml`` datasets.

        This schema should contain information about splits, target properties,
        and if relevant, graph wiring and so on.

        Returns
        -------
        DatasetSchema
            Validated ``DatasetSchema`` object

        Raises
        ------
        RuntimeError:
            If there is no metadata
        """
        if not hasattr(self, "_metadata"):
            meta_target = self.filepath.parent.joinpath("metadata.json")
            if not meta_target.exists():
                raise FileNotFoundError(
                    "No `metadata.json` specifying DatasetSchema found in dataset directory.."
                )
            with open(meta_target) as read_file:
                metadata = DatasetSchema.model_validate_json(
                    read_file.read(), strict=True
                )
            self._metadata = metadata
        return self._metadata

    @property
    @cache
    def blake2s_checksum(self) -> str:
        """
        Computes the BLAKE2s hash for the current dataset.

        This functions by opening the binary file for reading and
        iterating over lines in the file.

        Returns
        -------
        str
            BLAKE2s hash for the HDF5 file of this dataset.
        """
        with open(self.filepath, "rb") as read_file:
            hasher = blake2s()
            for line in read_file.readlines():
                hasher.update(line)
            return hasher.hexdigest()

    def read_data(self) -> h5py.File:
        return h5py.File(str(self.filepath.absolute()), mode="r")

    def write_data(
        self, index: int, sample: DataSampleSchema, overwrite: bool = False
    ) -> None:
        """
        Writes a data sample at index to the current HDF5 file.

        Most likely not the most performant way to write data
        since it's in serial, but is easily accessible.

        Parameters
        ----------
        index : int
            Index to write the data to. Must not already be
            present in the dataset if ``overwrite`` is False.
        sample : DataSampleSchema
            A data sample defined by an instance of ``DataSampleSchema``.
        overwrite : bool, default False
            If False, if ``index`` already exists in the file
            a ``RuntimeError`` will be raised.
        """
        with h5py.File(str(self.filepath).absolute(), "w") as h5_file:
            if overwrite and str(index) in h5_file:
                del h5_file[str(index)]
            group = h5_file.create_group(str(index))
            sample_data = sample.model_dump(round_trip=True)
            for key, value in sample_data.items():
                write_data_to_hdf5_group(key, value, group)

    @cache
    def __len__(self) -> int:
        with self.read_data() as h5_data:
            return len(h5_data.keys())

    @property
    @cache
    def keys(self) -> list[str]:
        with self.read_data() as h5_data:
            return list(h5_data.keys())

    def __getitem__(self, index: int):
        data_index = self.keys[index]
        with self.read_data() as h5_data:
            try:
                sample_group = h5_data[data_index]
            except KeyError as e:
                raise KeyError(f"Data sample {data_index} missing from dataset.") from e
            sample_data = {}
            for key, value in sample_group.items():
                sample_data[key] = value
            # validate expected data
            for target in self.metadata.targets:
                if target.name not in sample_data:
                    raise KeyError(
                        f"Expected {target.name} in data sample but not found."
                    )
            sample = DataSampleSchema(**sample_data)
            # now try to apply transforms
            if self.transforms:
                for transform in self.transforms:
                    try:
                        sample = transform(sample)
                    except Exception as e:
                        raise RuntimeError(
                            f"Unable to apply {transform} on sample at index {index}."
                        ) from e
            return sample


class MatSciMLDataModule(pl.LightningDataModule):
    def __init__(
        self,
        filepath: PathLike,
        transforms: list[Callable] | None = None,
        strict_checksum: bool = False,
        **loader_kwargs,
    ):
        """
        Initialize a ``MatSciMLDataModule`` that uses the HDF5
        binary data format. Provides a ``Lightning`` wrapper around
        the dataset class, which individually handles splits whereas
        this class handles a collection of HDF5 files.

        Parameters
        ----------
        filepath : PathLike
            Filepath to a root folder containing HDF5 files
            for each split, and a metadata JSON file.
        transforms : list[Callable], optional
            List of transforms to process data samples after loading.
        loader_kwargs
            Additional keyword arguments that are passed to
            dataloaders.

        Raises
        ------
        RuntimeError:
            If the provided filepath is not a directory, this method
            will raise a ``RuntimeError``.
        """
        super().__init__()
        if not isinstance(filepath, Path):
            filepath = Path(filepath)
        if not filepath.is_dir():
            raise RuntimeError(f"Expected filepath to be a directory; got {filepath}")
        self.metadata = filepath.joinpath("metadata.json")
        # add to things to save
        hparams_to_save = {
            "filepath": filepath,
            "transforms": transforms,
            "strict_checksum": strict_checksum,
            "metadata": self.metadata.model_dump(),
            "loader_kwargs": loader_kwargs,
        }
        self.save_hyperparameters(hparams_to_save)

    @property
    def metadata(self) -> DatasetSchema:
        return self._metadata

    @metadata.setter
    def metadata(self, filepath: Path) -> None:
        if not filepath.exists():
            raise RuntimeError(
                "No metadata found in target directory. Expected a metadata.json to exist."
            )
        with open(filepath, "r") as read_file:
            metadata = DatasetSchema.model_validate_json(read_file.read(), strict=True)
            self._metadata = metadata

    @property
    def h5_files(self) -> dict[Literal["train", "test", "validation", "predict"], Path]:
        """
        Returns a mapping of split to HDF5 filepaths within
        the root folder. Entries will only be present if the file
        can be found.

        Returns
        -------
        dict[Literal["train", "test", "validation", "predict"], Path]
            Available split to HDF5 filepath mapping.
        """
        return self._h5_files

    @h5_files.setter
    def h5_files(self, root_dir: Path) -> None:
        """
        Given a root folder directory, discover subsplits of data
        by matching the ``.h5`` file extension.

        Parameters
        ----------
        root_dir : Path
            Folder containing data splits and metadata.

        Raises
        ------
        RuntimeError
            If not ``.h5`` files were discovered within this folder,
            we raise a ``RuntimeError``.
        """
        h5_files = {}
        for prefix in ["train", "test", "validation", "predict"]:
            h5_file = root_dir.joinpath(prefix).with_suffix(".h5")
            if h5_file.exists():
                h5_files[prefix] = h5_file.absolute()
                logger.debug(f"Found {h5_file} data.")
        if len(h5_files) == 0:
            raise RuntimeError("No .h5 files found in target directory.")
        self._h5_files = h5_files

    def setup(self, stage: str):
        # check and set the available HDF5 data files
        self.h5_files = self.hparams.filepath
        self.datasets = {
            key: MatSciMLDataset(
                path, self.hparams.transforms, self.hparams.strict_checksum
            )
            for key, path in self.h5_files.items()
        }
        if stage == "fit":
            assert "train" in self.datasets, "No training split available!"
        if stage == "predict":
            assert "predict" in self.datasets, "No predict split available!"
