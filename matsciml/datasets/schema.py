from __future__ import annotations

from importlib import import_module
from enum import Enum
from datetime import datetime
from typing import Literal, Any
from os import PathLike
from pathlib import Path

import orjson
from ase import Atoms
from pydantic import BaseModel, ConfigDict, ValidationError, model_validator
from numpydantic import NDArray, Shape
from loguru import logger

from matsciml.datasets.utils import orjson_dumps
from matsciml.common.packages import package_registry

"""This module defines schemas pertaining to data, using ``pydantic`` models
to help with validation and (de)serialization.

The driving principle behind this is to try and define a standardized data
format that is also relatively low maintenance. The ``DatasetSchema`` is
used to fully qualify a dataset, including file hashing, a list of expected
targets, and so on. The ``DataSampleSchema`` provides an "on-the-rails"
experience for both developers and users by defining a consistent set of
attribute names that should be
"""

# ruff: noqa: F722


class DatasetEnum(str, Enum):
    s2ef = "S2EFDataset"
    is2re = "IS2REDataset"
    mp20 = "MaterialsProjectDataset"
    mptraj = "MaterialsTrajectoryDataset"
    alexandria = "AlexandriaDataset"
    nomad = "NomadDataset"
    oqmd = "OQMDDataset"
    generic = "GenericDataset"


class PeriodicBoundarySchema(BaseModel):
    """
    Specifies periodic boundary conditions for each axis.
    """

    x: bool
    y: bool
    z: bool


class NormalizationSchema(BaseModel):
    target_key: str
    mean: float
    std: float


class GraphWiringSchema(BaseModel):
    """
    Provides a specification for tracking how graphs within
    a dataset are wired. Primarily, a ``cutoff_radius`` is
    specified to artificially truncate a neighborhood, and
    a package or algorithm is used to compute this neighborhood
    function.

    The validation of this schema includes checking to ensure
    that the

    Attributes
    ----------
    cutoff_radius : float
        Cutoff radius used to specify the neighborhood region,
        typically assumed to be in angstroms for most algorithms.
    algorithm : Literal['pymatgen', 'ase', 'custom']
        Algorithm used for computing the atom neighborhoods and
        subsequently edges. If either ``pymatgen`` or ``ase`` are
        specified, schema validation will include checking versions.
    allow_mismatch : bool
        If set to True, we will not perform the algorithm version
        checking. If set to False, a mismatch in algorithm version
        will throw a ``ValidationError``.
    algo_version : str, optional
        Version number of ``pymatgen`` or ``ase`` depending on which
        is being used. If ``algorithm`` is 'custom', this is ignored
        as this check has not yet been implemented.
    algo_hash_path : str, optional
        Nominally a path to a Python import that, when imported, returns
        a hash used to match against ``algo_hash``. Currently not implemented.
    algo_hash : str, optional
        Version hash used for a custom algorithm. Currently not implemented.
    kwargs : dict[str, Any], optional
        Additional keyword arguments that might be passed
        to custom algorithms.
    """

    cutoff_radius: float
    algorithm: Literal["pymatgen", "ase", "custom"]
    allow_mismatch: bool
    algo_version: str | None = None
    algo_hash_path: str | None = None
    algo_hash: str | None = None
    max_neighbors: int = -1
    kwargs: dict[str, Any] | None = None

    @model_validator(mode="after")
    def check_algo_version(self):
        if not self.algo_version and not self.algo_hash:
            raise ValidationError(
                "At least one form of algorithm versioning is required."
            )
        if self.algo_version:
            if self.algorithm == "pymatgen":
                pmg = import_module("pymatgen.core")
                actual_version = pmg.__version__
            elif self.algorithm == "ase":
                ase = import_module("ase")
                actual_version = ase.__version__
            else:
                logger.warning(
                    "Using custom algorithm with version specified - checking is not currently supported."
                )
                return self
            # throw validation error only if we don't allow mismatches
            if self.algo_version != actual_version and not self.allow_mismatch:
                raise ValidationError(
                    f"GraphWiringSchema algorithm version mismatch for package {self.algorithm}"
                    f" installed {actual_version}, expected {self.algo_version}."
                )
            elif self.algo_version != actual_version:
                logger.warning(
                    f"GraphWiringSchema algorithm version mismatch for package {self.algorithm}"
                    f" installed {actual_version}, expected {self.algo_version}."
                    " `allow_mismatch` was set to True, turning this into a warning message instead of an exception."
                )
            return self
        if self.algo_hash:
            algo_path = getattr(self, "algo_hash_path", None)
            if not algo_path:
                raise ValidationError(
                    "Graph wiring algorithm hash specified but no path to resolve."
                )
            logger.warning(
                "Hash checking for custom algorithms is not currently implemented."
            )
            return self


class DatasetSchema(BaseModel):
    name: DatasetEnum
    creation: datetime
    target_keys: list[str]
    split_blake2s: set
    modified: datetime | None = None
    description: str | None = None
    graph_schema: GraphWiringSchema | None = None
    normalization: dict[str, NormalizationSchema] | None = None

    class Config:
        json_loads = orjson.loads
        json_dumps = orjson_dumps

    @classmethod
    def from_json(cls, json_path: PathLike) -> DataSampleSchema:
        """
        Deserialize a JSON file, validating against the expected dataset
        schema.

        Parameters
        ----------
        json_path : PathLike
            Path to a JSON metadata file.

        Returns
        -------
        DataSampleSchema
            Instance of a validated ``DatasetSchema``.

        Raises
        ------
        FileNotFoundError
            If the specified JSON file does not exist.
        """
        if not isinstance(json_path, Path):
            json_path = Path(json_path)
        if not json_path.exists():
            raise FileNotFoundError(
                f"{json_path} JSON metadata file could not be found."
            )
        with open(json_path, "r") as read_file:
            return cls.model_validate_json(read_file.read(), strict=True)

    @model_validator(mode="after")
    def check_target_normalization(self):
        if self.normalization is not None:
            # first check every key is available as targets
            target_keys = set(self.target_keys)
            norm_keys = set(self.normalization.keys())
            # check to see if we have unexpected norm keys
            diff = norm_keys - target_keys
            if len(diff) > 0:
                raise ValidationError(f"Unexpected keys in normalization: {diff}")
        return self


class DataSampleSchema(BaseModel):
    """
    Intention behind this schema is to have a superset of
    ``ase.Atoms``: a fully qualified description of an atomistic
    system including spin.
    """

    index: int
    num_atoms: int
    cart_coords: NDArray[Shape["*, 3"], float]
    atomic_numbers: NDArray[Shape["*"], int]
    pbc: PeriodicBoundarySchema
    electron_spins: NDArray[Shape["*"], float] | None = None  # electronic spin at atom
    nuclear_spins: NDArray[Shape["*"], float] | None = (
        None  # optional nuclear spin at atom
    )
    isotopic_masses: NDArray[Shape["*"], float] | None = None
    atomic_charges: NDArray[Shape["*"], float] | None = None
    atomic_energies: NDArray[Shape["*"], float] | None = None
    atomic_labels: NDArray[Shape["*"], int] | None = (
        None  # allows atoms to be tagged with class labels
    )
    total_energy: float | None = None
    forces: NDArray[Shape["*, 3"], float] | None = None
    stresses: NDArray[Shape["*, 3, 3"], float] | None = None
    lattice_parameters: NDArray[Shape["6"], float] | None = None
    lattice_matrix: NDArray[Shape["3, 3"], float] | None = None
    edge_index: NDArray[Shape["2, *"], int] | None = (
        None  # allows for precomputed edges
    )
    charge: float | None = None  # overall system charge
    multiplicity: float | None = None  # overall system multiplicity
    images: NDArray[Shape["*, 3"], int] | None = None
    offsets: NDArray[Shape["*, 3"], float] | None = None
    unit_offsets: NDArray[Shape["*, 3"], float] | None = None
    graph: Any = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def to_ase_atoms(self) -> Atoms:
        return Atoms(
            positions=self.cart_coords,
            cell=self.lattice_matrix,
            numbers=self.atomic_numbers,
            tags=self.atomic_labels,
            charges=self.atomic_charges,
            masses=self.isotopic_masses,
        )

    @property
    def graph_backend(self) -> Literal["dgl", "pyg"] | None:
        if not self.graph:
            return None
        else:
            if "pyg" in package_registry:
                from torch_geometric.data import Data as PyGGraph

                if isinstance(self.graph, PyGGraph):
                    return "pyg"
            elif "dgl" in package_registry:
                from dgl import DGLGraph

                if isinstance(self.graph, DGLGraph):
                    return "dgl"
            else:
                raise TypeError(f"Unexpected graph type: {type(self.graph)}")
