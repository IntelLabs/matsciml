# Copyright (C) 2022-3 Intel Corporation
# SPDX-License-Identifier: MIT License

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from pathlib import Path

# TODO homogenize the variable and folder names
devset_path = Path(__file__).parents[0].joinpath("dev-s2ef-dgl")
s2ef_devset = devset_path
is2re_devset = Path(__file__).parents[0].joinpath("dev-is2re-dgl")

from ocpmodels.datasets.base import PointCloudDataset
from ocpmodels.datasets.ocp_datasets import IS2REDataset, S2EFDataset
from ocpmodels.datasets.multi_dataset import MultiDataset
