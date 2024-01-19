# Copyright (C) 2022-3 Intel Corporation
# SPDX-License-Identifier: MIT License
from __future__ import annotations

from os import getenv
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import random_split

from matsciml.common.registry import registry
from matsciml.datasets import MultiDataset


@registry.register_datamodule("MatSciMLDataModule")
class MatSciMLDataModule(pl.LightningDataModule):
    r"""
    Initializes a `MatSciMLDataModule`, which is the primary Lightning Datamodule for
    interacting with single types of datasets.

    Parameters
    ----------
    dataset : Optional[Union[str, Type[TorchDataset], TorchDataset]], optional
        Reference to a dataset class or object instance. The former can be specified
        by providing the name of the dataset, providing it is contained in the registry,
        or by providing a reference to the class; in either case, it is subsequently
        used with paths provided to instantiate the class. Alternatively, an instance
        of the dataset can be passed directly.
    train_path : Optional[Union[str, Path]], optional
        Path to a training dataset, by default None
    batch_size : int, optional
        Number of data samples per batch, by default 32
    num_workers : int, optional
        Number of data loader workers, by default 0, which equates to
        only using the main process for data loading.
    val_split : Optional[Union[str, Path, float]], optional
        Split parameter used for validation, which can be a float between 0/1
         or a string/path, by default 0.0 which skips validation.
    test_split : Optional[Union[str, Path, float]], optional
        Split parameter used for test, which can be a float between 0/1
         or a string/path, by default 0.0 which skips validation.
    seed : Optional[int], optional
        Random seed value used to create splits if fractional values are
        passed into ``val_split``/``test_split``, by default None, which
        will first try to read the environment variable followed by using
        a hardcoded value.
    dset_kwargs : Optional[Dict[str, Any]], optional
        Kwargs passed into the construction of the dataset object, by default None

    Examples
    ----------
    There are three approaches to instantiate this class:

    1. Passing a string name or dataset type into the ``dataset`` argument:

    >>> datamodule = MatSciMLDataModule(dataset="MaterialsProjectDataset", train_path="/path/to/data/)
    >>> datamodule = MatSciMLDataModule(dataset=matsciml.datasets.MaterialsProjectDataset, ...)

    2. Using the ``from_devset`` class method:

    >>> datamodule = MatSciMLDataModule.from_devset(dataset="MaterialsProjectDataset")

    3. Passing a dataset directly into the ``dataset`` argument:

    >>> datamodule = MatSciMLDataModule(dataset=MaterialsProjectDataset("/path/to/train_data"))

    To specify val/test splits, you can pass a float or path/string: the former relies on passing
    ``train_path``, and extract out a fraction of it to use for that particular split. The latter
    will use dedicated data holdouts for splits:

    >>> datamodule = MatSciMLDataModule.from_devset(
            dataset="MaterialsProjectDataset",
            train_path="/path/to/train_data",
            val_split=0.2           # this uses 20% of the full data contained in ``train_path``
        )
    >>> datamodule = MatSciMLDataModule.from_devset(
            dataset="MaterialsProjectDataset",
            train_path="/path/to/train_data",
            val_split="/path/to/val_data"    # this uses a pre-determined split
        )

    To convert between formats (i.e. graphs vs. point clouds), include it as a transform by
    passing it into ``dset_kwargs``:

    >>> datamodule = MatSciMLDataModule.from_devset(
            dataset="MaterialsProjectDataset",
            dset_kwargs={
                "transforms": [PointCloudToGraphTransform(backend="dgl", cutoff=10.0)]
            },
        )
    """

    def __init__(
        self,
        dataset: str | type[TorchDataset] | TorchDataset | None = None,
        train_path: str | Path | None = None,
        batch_size: int = 32,
        num_workers: int = 0,
        val_split: str | Path | float | None = 0.0,
        test_split: str | Path | float | None = 0.0,
        seed: int | None = None,
        dset_kwargs: dict[str, Any] | None = None,
        persistent_workers: bool | None = None,
    ):
        super().__init__()
        # make sure we have something to work with
        assert any(
            [i for i in [dataset, train_path, val_split, test_split]],
        ), f"No splits provided to datamodule."
        # if floats are passed to splits, make sure dataset is provided for inference
        if any([isinstance(i, float) for i in [val_split, test_split]]):
            assert (
                dataset is not None
            ), f"Float passed to split, but no dataset provided to split."
        if isinstance(dataset, type):
            assert any(
                [
                    isinstance(p, (str, Path))
                    for p in [train_path, val_split, test_split]
                ],
            ), "Dataset type passed, but no paths to construct with."
        self.dataset = dataset
        self.dset_kwargs = dset_kwargs
        self.persistent_workers = persistent_workers
        self.save_hyperparameters(ignore=["dataset"])

    @property
    def persistent_workers(self) -> bool:
        """
        Flag to denote whether data loader workers are pinned or not.

        This property can be overridden by user by explicitly passing
        ``persistent_workers`` into the class constructor. Otherwise,
        the default behavior is just to have persistent workers if there
        ``num_workers`` > 0.

        Returns
        -------
        bool
            True if data loader workers are pinned, otherwise False
        """
        is_persist = getattr(self, "_persistent_workers", None)
        if is_persist is None:
            return self.hparams.num_workers > 0
        else:
            return is_persist

    @persistent_workers.setter
    def persistent_workers(self, value: None | bool) -> None:
        self._persistent_workers = value

    def _make_dataset(
        self,
        path: str | Path,
        dataset: TorchDataset | type[TorchDataset],
    ) -> TorchDataset:
        """
        Convert a string or path specification of a dataset into a concrete dataset object.

        Parameters
        ----------
        spec : Union[str, Path]
            String or path to data split
        dataset : Union[Torch.Dataset, Type[TorchDataset]]
            A dataset object or type. If the former, the transforms from this dataset
            will be copied over to be applied to the new split.

        Returns
        -------
        TorchDataset
            Dataset corresponding to the given path
        """
        dset_kwargs = getattr(self, "dset_kwargs", None)
        if not dset_kwargs:
            dset_kwargs = {}
        # try and grab the dataset class from registry
        if isinstance(dataset, str):
            dset_string = dataset
            dataset = registry.get_dataset_class(dataset)
            if not dataset:
                valid_keys = registry.__entries__["datasets"].keys()
                raise KeyError(
                    f"Incorrect dataset specification from string: passed {dset_string}, but not found in registry: {valid_keys}.",
                )
        if isinstance(dataset, TorchDataset):
            transforms = getattr(dataset, "transforms", None)
            dset_kwargs["transforms"] = transforms
            # apply same transforms to this split
            new_dset = dataset.__class__(path, **dset_kwargs)
        else:
            new_dset = dataset(path, **dset_kwargs)
        return new_dset

    def setup(self, stage: str | None = None) -> None:
        splits = {}
        # set up the training split, if provided
        if getattr(self.hparams, "train_path", None) is not None:
            train_dset = self._make_dataset(self.hparams.train_path, self.dataset)
            # set the main dataset to the train split, since it's used for other splits
            self.dataset = train_dset
            splits["train"] = train_dset
        # now make test and validation splits. If both are floats, we'll do a joint split
        if any(
            [
                isinstance(self.hparams[key], float)
                for key in ["val_split", "test_split"]
            ],
        ):
            # in the case that floats are provided for
            if self.hparams.seed is None:
                # try read from PyTorch Lightning, if not use a set seed
                seed = getenv("PL_GLOBAL_SEED", 42)
            else:
                seed = self.hparams.seed
            generator = torch.Generator().manual_seed(int(seed))
            num_points = len(self.dataset)
            # grab the fractional splits, but ignore them if they are not floats
            val_split = getattr(self.hparams, "val_split")
            if not isinstance(val_split, float):
                val_split = 0.0
            test_split = getattr(self.hparams, "test_split")
            if not isinstance(test_split, float):
                test_split = 0.0
            num_val = int(val_split * num_points)
            num_test = int(test_split * num_points)
            # make sure we're not asking for more data than exists
            num_train = num_points - (num_val + num_test)
            assert (
                num_train >= 0
            ), f"More test/validation samples requested than available samples."
            splits_list = random_split(
                self.dataset,
                [num_train, num_val, num_test],
                generator,
            )
            for split, key in zip(splits_list, ["train", "val", "test"]):
                if split is not None:
                    splits[key] = split
        # otherwise, just assume paths - if they're not we'll ignore them here
        for key in ["val", "test"]:
            split_path = getattr(self.hparams, f"{key}_split", None)
            if isinstance(split_path, (str, Path)):
                dset = self._make_dataset(split_path, self.dataset)
                splits[key] = dset
        # the last case assumes only the dataset is passed, we will treat it as train
        if len(splits) == 0:
            splits["train"] = self.dataset
        self.splits = splits

    def train_dataloader(self):
        split = self.splits.get("train")
        return DataLoader(
            split,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            collate_fn=self.dataset.collate_fn,
            persistent_workers=self.persistent_workers,
        )

    def predict_dataloader(self):
        """
        Predict behavior just assumes the whole dataset is used for inference.
        """
        return DataLoader(
            self.dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            collate_fn=self.dataset.collate_fn,
            persistent_workers=self.persistent_workers,
        )

    def test_dataloader(self):
        split = self.splits.get("test", None)
        if split is None:
            return None
        return DataLoader(
            split,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            collate_fn=self.dataset.collate_fn,
            persistent_workers=self.persistent_workers,
        )

    def val_dataloader(self):
        split = self.splits.get("val", None)
        if split is None:
            return None
        return DataLoader(
            split,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            collate_fn=self.dataset.collate_fn,
            persistent_workers=self.persistent_workers,
        )

    @classmethod
    def from_devset(
        cls,
        dataset: str,
        dset_kwargs: dict[str, Any] = {},
        **kwargs,
    ):
        r"""
        Instantiate a data module from a dataset's devset.

        This is intended mostly for testing and debugging purposes, with the
        bare number of args/kwargs required to get up and running. The behavior
        of this method will replicate the devset for train, validation, and test
        to allow each part of the pipeline to be tested.

        Parameters
        ----------
        dataset : str
            Class name for dataset to use
        dset_kwargs : Dict[str, Any], optional
            Dictionary of keyword arguments to be passed into
            the dataset creation, for example 'transforms', by default {}

        Returns
        -------
        MatSciMLDataModule
            Instance of `MatSciMLDataModule` from devset

        Raises
        ------
        NotImplementedError
            If the dataset specified does not contain a devset path, this
            method will raise 'NotImplementedError'.
        """
        kwargs.setdefault("batch_size", 8)
        kwargs.setdefault("num_workers", 0)
        dset_kwargs.setdefault("transforms", None)
        dset = registry.get_dataset_class(dataset)
        devset_path = getattr(dset, "__devset__", None)
        if not devset_path:
            raise NotImplementedError(
                f"Dataset {dset.__name__} does not contain a '__devset__' attribute, cannot instantiate from devset.",
            )
        datamodule = cls(
            dset,
            train_path=devset_path,
            val_split=devset_path,
            test_split=devset_path,
            dset_kwargs=dset_kwargs,
            **kwargs,
        )
        return datamodule

    @property
    def target_keys(self) -> dict[str, list[str]]:
        return self.dataset.target_keys


@registry.register_datamodule("MultiDataModule")
class MultiDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int = 32,
        num_workers: int = 0,
        train_dataset: MultiDataset | None = None,
        val_dataset: MultiDataset | None = None,
        test_dataset: MultiDataset | None = None,
        predict_dataset: MultiDataset | None = None,
        persistent_workers: bool | None = None,
    ) -> None:
        r"""
        Data module specifically for using multiple different datasets in tandem.

        Parameters
        ----------
        batch_size : int, optional
            Number of **total** data samples contained in a batch, comprising
            a mix of samples from each dataset, by default 32
        num_workers : int, optional
            Number of parallel data loader workers, by default 0
        train_dataset : Optional[MultiDataset], optional
            Instance of ``MultiDataset`` to use for training, by default None
        val_dataset : Optional[MultiDataset], optional
            Instance of ``MultiDataset`` to use for validation, by default None
        test_dataset : Optional[MultiDataset], optional
            Instance of ``MultiDataset`` to use for testing, by default None
        predict_dataset : Optional[MultiDataset], optional
            Instance of ``MultiDataset`` to use for inference, by default None

        Examples
        ------
        The class can be instantiated with one or more datasets passed into each
        split. The configuration is slightly inconvenient, owing to the fact that
        there is a lot to specify:

        Train on IS2RE, S2EF, and Materials Project:

        >>> datamodule = MultiDataModule(
                train_dataset=MultiDataset(
                    [
                        IS2REDataset("/path/to/is2re"),
                        S2EFDataset("/path/to/s2ef"),
                        MaterialsProjectDataset("/path/to/mp_data")
                    ]
                )
            )

        Train on IS2RE+S2EF, validate on LiPS+S2EF (for pedagogical reasons):

        >>> datamodule = MultiDataModule(
                train_dataset=MultiDataset(
                    [
                        IS2REDataset("/path/to/is2re"),
                        S2EFDataset("/path/to/s2ef")
                    ]
                ),
                val_dataset=MultiDataset(
                    [
                        S2EFDataset("/path/to/another/s2ef"),
                        LiPSDataset("/path/to/lips")
                    ]
                )
            )
        """
        super().__init__()
        if not any([train_dataset, val_dataset, test_dataset, predict_dataset]):
            raise ValueError(
                f"No datasets were passed for training, validation, testing, or predict.",
            )
        self.save_hyperparameters(
            ignore=["train_dataset", "val_dataset", "test_dataset", "predict_dataset"],
        )
        # stash the datasets as an attribute
        self.datasets = {
            key: value
            for key, value in zip(
                ["train", "val", "test", "predict"],
                [train_dataset, val_dataset, test_dataset, predict_dataset],
            )
        }
        self.persistent_workers = persistent_workers

    @property
    def persistent_workers(self) -> bool:
        """
        Flag to denote whether data loader workers are pinned or not.

        This property can be overridden by user by explicitly passing
        ``persistent_workers`` into the class constructor. Otherwise,
        the default behavior is just to have persistent workers if there
        ``num_workers`` > 0.

        Returns
        -------
        bool
            True if data loader workers are pinned, otherwise False
        """
        is_persist = getattr(self, "_persistent_workers", None)
        if is_persist is None:
            return self.hparams.num_workers > 0
        else:
            return is_persist

    @persistent_workers.setter
    def persistent_workers(self, value: None | bool) -> None:
        self._persistent_workers = value

    @property
    def target_keys(self) -> dict[str, dict[str, list[str]]]:
        return self.datasets["train"].target_keys

    # Cannot return None for dataloader
    # https://github.com/Lightning-AI/pytorch-lightning/issues/15703#issuecomment-1872664346
    def train_dataloader(self) -> DataLoader | list[Any]:
        loader = []
        data = self.datasets.get("train", None)
        if data:
            loader = DataLoader(
                data,
                self.hparams.batch_size,
                num_workers=self.hparams.num_workers,
                shuffle=True,
                collate_fn=data.collate_fn,
                persistent_workers=self.persistent_workers,
            )
        return loader

    def val_dataloader(self) -> DataLoader | list[Any]:
        loader = []
        data = self.datasets.get("val", None)
        if data:
            loader = DataLoader(
                data,
                self.hparams.batch_size,
                num_workers=self.hparams.num_workers,
                collate_fn=data.collate_fn,
                persistent_workers=self.persistent_workers,
            )
        return loader

    def test_dataloader(self) -> DataLoader | list[Any]:
        loader = []
        data = self.datasets.get("test", None)
        if data:
            loader = DataLoader(
                data,
                self.hparams.batch_size,
                num_workers=self.hparams.num_workers,
                collate_fn=data.collate_fn,
                persistent_workers=self.persistent_workers,
            )
        return loader

    def predict_dataloader(self) -> DataLoader | list[Any]:
        loader = []
        data = self.datasets.get("predict", None)
        if data:
            loader = DataLoader(
                data,
                self.hparams.batch_size,
                num_workers=self.hparams.num_workers,
                collate_fn=data.collate_fn,
                persistent_workers=self.persistent_workers,
            )
        return loader

