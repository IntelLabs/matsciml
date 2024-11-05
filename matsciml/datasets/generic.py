from __future__ import annotations

from functools import cache
from os import PathLike
from pathlib import Path
from typing import Callable, Literal
from logging import getLogger

import h5py
from torch.utils.data import Dataset
from lightning import pytorch as pl

from matsciml.datasets.schema import DatasetSchema, DataSampleSchema


logger = getLogger("matsciml.datasets.MatSciMLDataset")


class MatSciMLDataset(Dataset):
    def __init__(self, filepath: PathLike, transforms: list[Callable] | None = None):
        if not isinstance(filepath, Path):
            filepath = Path(filepath)
        self.filepath = filepath
        self.transforms = transforms

    @property
    def metadata(self) -> DatasetSchema:
        meta_target = self.filepath.parent.joinpath("metadata.json")
        if not meta_target.exists():
            raise RuntimeError("No metadata for dataset.")
        return DatasetSchema.parse_file(meta_target)

    @property
    def data(self) -> h5py.File:
        return h5py.File(str(self.filepath.absolute()), mode="r")

    @cache
    def __len__(self) -> int:
        with self.data as h5_data:
            return len(h5_data.keys())

    def __getitem__(self, index: int):
        index = str(index)
        with self.data as h5_data:
            try:
                sample_group = h5_data[index]
            except KeyError as e:
                raise KeyError(f"Data sample {index} missing from dataset.") from e
            sample_data = {}
            for key, value in sample_group.items():
                sample_data[key] = value
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
        **loader_kwargs,
    ):
        """
        Initialize a ``MatSciMLDataModule`` that uses the HDF5
        binary data format.

        [TODO:description]

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
        if not isinstance(filepath, Path):
            filepath = Path(filepath)
        if not filepath.is_dir():
            raise RuntimeError(f"Expected filepath to be a directory; got {filepath}")
        self.metadata = filepath.joinpath("metadata.json")
        self.filepath = filepath
        self.transforms = transforms
        self.loader_kwargs = loader_kwargs

    @property
    def metadata(self) -> DatasetSchema:
        return self._metadata

    @metadata.setter
    def metadata(self, filepath: Path) -> None:
        if not filepath.exists():
            raise RuntimeError(
                "No metadata found in target directory. Expected a metadata.json to exist."
            )
        metadata = DatasetSchema.parse_file(filepath)
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
        self.h5_files = self.filepath
        self.datasets = {
            key: MatSciMLDataset(path, self.transforms)
            for key, path in self.h5_files.items()
        }
        if stage == "fit":
            assert "train" in self.datasets, "No training split available!"
        if stage == "predict":
            assert "predict" in self.datasets, "No predict split available!"