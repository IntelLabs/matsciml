from __future__ import annotations

from enum import Enum
from datetime import datetime

from numpy import typing as nt
from pydantic import BaseModel


class DatasetEnum(str, Enum):
    s2ef = "S2EFDataset"
    is2re = "IS2REDataset"
    mp20 = "MaterialsProjectDataset"
    mptraj = "MaterialsTrajectoryDataset"
    alexandria = "AlexandriaDataset"
    nomad = "NomadDataset"
    oqmd = "OQMDDataset"
    generic = "GenericDataset"


class DatasetSchema(BaseModel):
    name: DatasetEnum
    creation: datetime
    blake2s: str
    target_keys: list[str]
    train_indices: list[int] | None = None
    val_indices: list[int] | None = None
    test_indices: list[int] | None = None
    predict_indices: list[int] | None = None
    modified: datetime | None = None


class DataSampleSchema(BaseModel):
    index: int
    cart_coords: nt.ArrayLike
    atomic_numbers: nt.ArrayLike
    forces: nt.ArrayLike | None = None
    stresses: nt.ArrayLike | None = None
    lattice_parameters: nt.ArrayLike | None = None
    lattice_matrix: nt.ArrayLike | None = None
