# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: MIT License

from abc import abstractclassmethod
from typing import Union, Optional, Type, List, Callable, Dict
from pathlib import Path
from warnings import warn
from os import getenv

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset as TorchDataset
from torch.utils.data import random_split

from ocpmodels.datasets import (
    IS2REDataset,
    S2EFDataset,
    PointCloudDataset,
    MultiDataset,
)
from ocpmodels.datasets import s2ef_devset, is2re_devset
from ocpmodels.datasets.materials_project import (
    materialsproject_devset,
    MaterialsProjectDataset,
    DGLMaterialsProjectDataset,
)
from ocpmodels.datasets.lips import LiPSDataset, DGLLiPSDataset, lips_devset


class GraphDataModule(pl.LightningDataModule):

    """
    A high level interface to OCP datasets. This encapsulates setting up of
    data set and data loaders, in a way that is backend agnostic. The choice
    of backend, either PyTorch Geometric ("pyg") or Deep Graph Library ("dgl")
    is deferred to class methods, and all the rest of the set up is designed
    to be abstract.

    Each class is initialized by passing a data loader (i.e. subclasses of
    `torch.utils.data.DataLoader`), and a dataset (e.g. `TrajectoryLMDB`).
    The main purpose of this class is to manage data splits, and define
    loading/transforms for each split. The initialization takes the paths
    to each split - as you would pass to the dataset - and will set up
    the dataset and loaders.

    PyTorch Lightning will handle wrapping `DataParallel` and `DDP` around
    the loaders when the `Trainer` strategy is set to the appropriate value.
    """

    _backend = None

    def __init__(
        self,
        train_path: Optional[str],
        dataset_class: Type[TorchDataset],
        batch_size: int = 32,
        num_workers: int = 0,
        val_path: Optional[str] = None,
        test_path: Optional[str] = None,
        predict_path: Optional[str] = None,
        transforms: Optional[List[Callable]] = None,
    ):
        super().__init__()
        self.paths = {
            "train": train_path,
            "val": val_path,
            "test": test_path,
            "predict": predict_path,
        }
        # check that the path is accessible first and not none
        self.verify_paths()
        assert (
            len(self.paths) > 0
        ), "No data paths were provided or valid; check configuration."
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset_class = dataset_class
        self.collate_fn = dataset_class.collate_fn
        self.transforms = transforms
        self.save_hyperparameters(ignore=["dataset_class"])

    def verify_paths(self) -> None:
        """
        Mutates the path dictionary in-place by iteratively checking the existence of
        each _specified_ path. The user is warned if the path is unresolvable with a
        quasi-informative warning, and appends the key to a `bad_key` list if it is
        unresolvable or None. Keys in this list are then removed from the dictionary
        in-place.
        """
        bad_keys = []
        for key, path in self.paths.items():
            # we only check specified paths, and drop them otherwise
            if path is not None:
                if isinstance(path, str):
                    path = Path(path)
                if not path.exists():
                    warn(
                        f"A path for {key} dataset was specified but unresolvable, please check {path.absolute()} exists and contains *.lmdb files."
                    )
                    bad_keys.append(key)
            else:
                bad_keys.append(key)
        for key in bad_keys:
            del self.paths[key]

    def setup(self, stage: Union[str, None] = None) -> None:
        """
        This is a PyTorch Lightning internal method, so one doesn't need
        to call this manually (normally). Here, it defines a dictionary
        of `data_splits`, which are then independently called upon in
        respective methods.

        Parameters
        ----------
        stage : Union[str, None], optional
            _description_, by default None
        """
        self.data_splits = {}
        # set up each of the dataset splits
        for key, path in self.paths.items():
            self.data_splits[key] = self.dataset_class(path, transforms=self.transforms)

    @property
    def target_keys(self) -> Dict[str, List[str]]:
        splits = getattr(self, "data_splits", None)
        if splits is None:
            self.setup()
            splits = self.data_splits
        # get the first dataset we can get
        dset = list(splits.values())[0]
        return dset.target_keys

    def train_dataloader(self):
        split = self.data_splits.get("train")
        return split.data_loader(
            split,
            shuffle=False,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self):
        split = self.data_splits.get("test")
        if split is not None:
            return split.data_loader(
                split,
                shuffle=False,
                num_workers=self.num_workers,
                batch_size=self.batch_size,
                collate_fn=self.collate_fn,
            )
        else:
            warn(f"Test split not defined; not performing testing.")
            pass

    def val_dataloader(self):
        split = self.data_splits.get("val")
        if split is not None:
            return split.data_loader(
                split,
                shuffle=False,
                num_workers=self.num_workers,
                batch_size=self.batch_size,
                collate_fn=self.collate_fn,
            )
        else:
            warn(f"Validation split not defined; not performing validation.")
            pass

    def predict_dataloader(self):
        split = self.data_splits.get("predict")
        if split is not None:
            return split.data_loader(
                split,
                shuffle=False,
                num_workers=self.num_workers,
                batch_size=self.batch_size,
                collate_fn=self.collate_fn,
            )


class S2EFDGLDataModule(GraphDataModule):
    """The DGL version of the S2EF task `LightningDataModule`"""

    def __init__(
        self,
        train_path: Optional[str] = None,
        batch_size: int = 32,
        num_workers: int = 0,
        val_path: Optional[str] = None,
        test_path: Optional[str] = None,
        predict_path: Optional[str] = None,
        transforms: Optional[List[Callable]] = None,
    ):
        super().__init__(
            train_path,
            S2EFDataset,
            batch_size,
            num_workers,
            val_path,
            test_path,
            predict_path,
            transforms,
        )

    @classmethod
    def from_devset(cls, **kwargs):
        kwargs.setdefault("batch_size", 8)
        kwargs.setdefault("num_workers", 0)
        return cls(s2ef_devset, **kwargs)


class IS2REDGLDataModule(GraphDataModule):
    """The DGL version of the IS2RE task `LightningDataModule`"""

    def __init__(
        self,
        train_path: Optional[str] = None,
        batch_size: int = 32,
        num_workers: int = 0,
        val_path: Optional[str] = None,
        test_path: Optional[str] = None,
        predict_path: Optional[str] = None,
        transforms: Optional[List[Callable]] = None,
    ):
        super().__init__(
            train_path,
            IS2REDataset,
            batch_size,
            num_workers,
            val_path,
            test_path,
            predict_path,
            transforms,
        )

    @classmethod
    def from_devset(cls, **kwargs):
        kwargs.setdefault("batch_size", 8)
        kwargs.setdefault("num_workers", 0)
        return cls(is2re_devset, **kwargs)


class DGLDataModule(S2EFDGLDataModule):
    """
    This class is implemented only to facilitate some backwards
    compatibility. The user is recommended to use the task specific
    data modules above.
    """

    def __init__(
        self,
        train_path: Optional[str] = None,
        batch_size: int = 32,
        num_workers: int = 0,
        val_path: Optional[str] = None,
        test_path: Optional[str] = None,
        predict_path: Optional[str] = None,
        transforms: Optional[List[Callable]] = None,
    ):
        super().__init__(
            train_path,
            batch_size,
            num_workers,
            val_path,
            test_path,
            predict_path,
            transforms,
        )
        warn(f"DGLDataModule is being retired - please switch to S2EFDGLDataModule.")


class BaseLightningDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset: Optional[Union[Type[TorchDataset], TorchDataset]] = None,
        train_path: Optional[Union[str, Path]] = None,
        batch_size: int = 32,
        num_workers: int = 0,
        val_split: Optional[Union[str, Path, float]] = 0.0,
        test_split: Optional[Union[str, Path, float]] = 0.0,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.dataset = dataset
        self.save_hyperparameters(ignore=["dataset"])

    def setup(self, stage: Optional[str] = None) -> None:
        if self.hparams.seed is None:
            # try read from PyTorch Lightning, if not use a set seed
            seed = getenv("PL_GLOBAL_SEED", 42)
        else:
            seed = self.hparams.seed
        generator = torch.Generator().manual_seed(int(seed))
        num_points = len(self.dataset)
        num_val = int(self.hparams.val_split * num_points)
        num_test = int(self.hparams.test_split * num_points)
        # make sure we're not asking for more data than exists
        num_train = num_points - (num_val + num_test)
        assert (
            num_train >= 0
        ), f"More test/validation samples requested than available samples."
        splits = random_split(self.dataset, [num_train, num_val, num_test], generator)
        self.splits = {key: splits[i] for i, key in enumerate(["train", "val", "test"])}

    def train_dataloader(self):
        split = self.splits.get("train")
        return DataLoader(
            split,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            collate_fn=self.dataset.collate_fn,
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
        )

    def test_dataloader(self):
        split = self.splits.get("test")
        if len(split) == 0:
            return None
        return DataLoader(
            split,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            collate_fn=self.dataset.collate_fn,
        )

    def val_dataloader(self):
        split = self.splits.get("val")
        if len(split) == 0:
            return None
        return DataLoader(
            split,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            collate_fn=self.dataset.collate_fn,
        )

    @abstractclassmethod
    def from_devset(cls, *args, **kwargs):
        ...

    @property
    def target_keys(self) -> Dict[str, List[str]]:
        return self.dataset.target_keys


class MaterialsProjectDataModule(BaseLightningDataModule):
    @classmethod
    def from_devset(
        cls, graphs: bool = True, transforms: Optional[List[Callable]] = None, **kwargs
    ):
        kwargs.setdefault("batch_size", 8)
        kwargs.setdefault("num_workers", 0)
        dset_class = (
            MaterialsProjectDataset if not graphs else DGLMaterialsProjectDataset
        )
        return cls(dset_class(materialsproject_devset, transforms=transforms), **kwargs)


class LiPSDataModule(BaseLightningDataModule):
    @classmethod
    def from_devset(
        cls, graphs: bool = True, transforms: Optional[List[Callable]] = None, **kwargs
    ):
        kwargs.setdefault("batch_size", 8)
        kwargs.setdefault("num_workers", 0)
        dset_class = LiPSDataset if not graphs else DGLLiPSDataset
        return cls(dset_class(lips_devset, transforms=transforms), **kwargs)


class PointCloudDataModule(GraphDataModule):
    def __init__(
        self,
        dataset_class: Type[TorchDataset],
        train_path: Optional[str] = None,
        batch_size: int = 32,
        num_workers: int = 0,
        point_cloud_size: int = 6,
        sample_size: int = 10,
        val_path: Optional[str] = None,
        test_path: Optional[str] = None,
        predict_path: Optional[str] = None,
        transforms: Optional[List[Callable]] = None,
    ):
        super().__init__(
            train_path,
            dataset_class,
            batch_size,
            num_workers,
            val_path,
            test_path,
            predict_path,
            transforms,
        )
        self._point_cloud_size = point_cloud_size
        self._sample_size = sample_size
        self.collate_fn = PointCloudDataset.collate_fn

    def setup(self, stage: Union[str, None] = None) -> None:
        """
        This class modifies the base behavior slightly, by wrapping the base
        dataset with `PointCloudDataset`.
        """
        self.data_splits = {}
        # set up each of the dataset splits
        for key, path in self.paths.items():
            self.data_splits[key] = PointCloudDataset(
                self.dataset_class(path), self._point_cloud_size, self._sample_size
            )

    @classmethod
    def from_s2ef(cls, **kwargs):
        """
        Convenient method to instantiate a `PointCloudDataModule` using
        the S2EF dataset. Kwargs are passed into the constructor
        method, and this method only overrides the dataset class explicitly.

        Returns
        -------
        PointCloudDataModule
            A point cloud lightning module configured to use the S2EF data.
        """
        return cls(dataset_class=S2EFDataset, **kwargs)

    @classmethod
    def from_is2re(cls, **kwargs):
        """
        Convenient method to instantiate a `PointCloudDataModule` using
        the S2EF dataset. Kwargs are passed into the constructor
        method, and this method only overrides the dataset class explicitly.

        Returns
        -------
        PointCloudDataModule
            A point cloud lightning module configured to use the IS2RE data.
        """
        return cls(dataset_class=IS2REDataset, **kwargs)


class MultiDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int = 32,
        num_workers: int = 0,
        train_dataset: Optional[MultiDataset] = None,
        val_dataset: Optional[MultiDataset] = None,
        test_dataset: Optional[MultiDataset] = None,
        predict_dataset: Optional[MultiDataset] = None,
    ) -> None:
        super().__init__()
        if not any([train_dataset, val_dataset, test_dataset, predict_dataset]):
            raise ValueError(
                f"No datasets were passed for training, validation, testing, or predict."
            )
        self.save_hyperparameters(
            ignore=["train_dataset", "val_dataset", "test_dataset", "predict_dataset"]
        )
        # stash the datasets as an attribute
        self.datasets = {
            key: value
            for key, value in zip(
                ["train", "val", "test", "predict"],
                [train_dataset, val_dataset, test_dataset, predict_dataset],
            )
        }

    @property
    def target_keys(self) -> Dict[str, Dict[str, List[str]]]:
        return self.datasets["train"].target_keys

    def train_dataloader(self) -> Union[DataLoader, None]:
        data = self.datasets.get("train", None)
        if data is None:
            return None
        return DataLoader(
            data,
            self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True,
            collate_fn=data.collate_fn,
        )

    def val_dataloader(self) -> Union[DataLoader, None]:
        data = self.datasets.get("val", None)
        if data is None:
            return None
        return DataLoader(
            data,
            self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            collate_fn=data.collate_fn,
        )

    def test_dataloader(self) -> Union[DataLoader, None]:
        data = self.datasets.get("test", None)
        if data is None:
            return None
        return DataLoader(
            data,
            self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            collate_fn=data.collate_fn,
        )

    def predict_dataloader(self) -> Union[DataLoader, None]:
        data = self.datasets.get("predict", None)
        if data is None:
            return None
        return DataLoader(
            data,
            self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            collate_fn=data.collate_fn,
        )
