# Copyright (C) 2022-3 Intel Corporation
# SPDX-License-Identifier: MIT License
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

from pathlib import Path

s2ef_devset = Path(__file__).parents[0].joinpath("dev-s2ef")
is2re_devset = Path(__file__).parents[0].joinpath("dev-is2re")


from matsciml.datasets.carolina_db import CMDataset
from matsciml.datasets.colabfit import ColabFitDataset
from matsciml.datasets.lips import LiPSDataset, lips_devset
from matsciml.datasets.materials_project import (
    MaterialsProjectDataset,
    materialsproject_devset,
)
from matsciml.datasets.multi_dataset import MultiDataset
from matsciml.datasets.nomad import NomadDataset
from matsciml.datasets.ocp_datasets import IS2REDataset, S2EFDataset
from matsciml.datasets.oqmd import OQMDDataset
from matsciml.datasets.symmetry import SyntheticPointGroupDataset, symmetry_devset

__all__ = [
    "IS2REDataset",
    "S2EFDataset",
    "CMDataset",
    "NomadDataset",
    "OQMDDataset",
    "MaterialsProjectDataset",
    "LiPSDataset",
    "SyntheticPointGroupDataset",
    "MultiDataset",
    "s2ef_devset",
    "is2redevset",
    "carolinadb_devset",
    "nomad_devset",
    "oqmd_devset",
    "materialsproject_devset",
    "lips_devset",
    "symmetry_devset",
    "ColabFitDataset",
]
