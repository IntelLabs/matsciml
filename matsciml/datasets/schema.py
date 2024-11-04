from __future__ import annotations

from enum import Enum
from datetime import datetime

from ase import Atoms
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
    splits: set
    modified: datetime | None = None
    description: str | None = None


class DataSampleSchema(BaseModel):
    """
    Intention behind this schema is to have a superset of
    ``ase.Atoms``: a fully qualified description of an atomistic
    system including spin.
    """

    index: int
    num_atoms: int
    cart_coords: nt.ArrayLike
    atomic_numbers: nt.ArrayLike
    electron_spins: nt.ArrayLike | None = None  # electronic spin at atom
    nuclear_spins: nt.ArrayLike | None = None  # optional nuclear spin at atom
    isotopic_masses: nt.ArrayLike | None = None
    atomic_charges: nt.ArrayLike | None = None
    atomic_energies: nt.ArrayLike | None = None
    atomic_labels: nt.ArrayLike | None = None  # allows atoms to be tagged
    total_energy: float | None = None
    forces: nt.ArrayLike | None = None
    stresses: nt.ArrayLike | None = None
    lattice_parameters: nt.ArrayLike | None = None
    lattice_matrix: nt.ArrayLike | None = None
    edge_index: nt.ArrayLike | None = None  # allows for precomputed edges
    charge: float | None = None  # overall system charge
    multiplicity: float | None = None  # overall system multiplicity
    images: nt.ArrayLike | None = None
    offsets: nt.ArrayLike | None = None
    unit_offsets: nt.ArrayLike | None = None

    def to_ase_atoms(self) -> Atoms:
        return Atoms(
            positions=self.cart_coords,
            cell=self.lattice_matrix,
            numbers=self.atomic_numbers,
            tags=self.atomic_labels,
            charges=self.atomic_charges,
            masses=self.isotopic_masses,
        )
