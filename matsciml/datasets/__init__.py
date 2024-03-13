# Copyright (C) 2022-3 Intel Corporation
# SPDX-License-Identifier: MIT License
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations


from matsciml.datasets.alexandria import AlexandriaDataset
from matsciml.datasets.carolina_db import CMDataset
from matsciml.datasets.colabfit import ColabFitDataset
from matsciml.datasets.lips import LiPSDataset
from matsciml.datasets.materials_project import MaterialsProjectDataset
from matsciml.datasets.multi_dataset import MultiDataset
from matsciml.datasets.nomad import NomadDataset
from matsciml.datasets.ocp_datasets import IS2REDataset, S2EFDataset
from matsciml.datasets.oqmd import OQMDDataset
from matsciml.datasets.symmetry import SyntheticPointGroupDataset

__all__ = [
    "AlexandriaDataset",
    "IS2REDataset",
    "S2EFDataset",
    "CMDataset",
    "NomadDataset",
    "OQMDDataset",
    "MaterialsProjectDataset",
    "LiPSDataset",
    "SyntheticPointGroupDataset",
    "MultiDataset",
    "ColabFitDataset",
]
