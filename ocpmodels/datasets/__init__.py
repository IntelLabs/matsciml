# Copyright (C) 2022-3 Intel Corporation
# SPDX-License-Identifier: MIT License

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from pathlib import Path

s2ef_devset = Path(__file__).parents[0].joinpath("dev-s2ef-dgl")
is2re_devset = Path(__file__).parents[0].joinpath("dev-is2re-dgl")


from ocpmodels.datasets.ocp_datasets import IS2REDataset, S2EFDataset
from ocpmodels.datasets.materials_project import MaterialsProjectDataset
from ocpmodels.datasets.lips import LiPSDataset
from ocpmodels.datasets.symmetry import SyntheticPointGroupDataset
from ocpmodels.datasets.multi_dataset import MultiDataset

__all__ = [
    "IS2REDataset",
    "S2EFDataset",
    "MaterialsProjectDataset",
    "LiPSDataset",
    "SyntheticPointGroupDataset",
    "MultiDataset",
]
