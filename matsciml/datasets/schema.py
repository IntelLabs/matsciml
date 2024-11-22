from __future__ import annotations

from importlib import import_module
from enum import Enum
from datetime import datetime
from typing import Literal, Any, Self
from os import PathLike
from pathlib import Path
import re

from ase import Atoms
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
    ValidationInfo,
)
from numpydantic import NDArray, Shape
from loguru import logger

from matsciml.common.packages import package_registry
from matsciml.common.inspection import get_all_args
from matsciml.modules.normalizer import Normalizer
from matsciml.datasets.transforms import PeriodicPropertiesTransform

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


class DataSampleEnum(str, Enum):
    scf = "SCFCycle"
    opt_trajectory = "OptimizationCycle"
    property = "Property"


class SplitHashSchema(BaseModel):
    """
    Schema for defining a set of data splits, with associated
    hashes for each split.

    This model will do a rudimentary check to make sure each
    value resembles a 64-character long hash. This is intended
    to work in tandem with the ``MatSciMLDataset.blake2s_checksum``
    property. For producers of datasets, you will need to be able
    to load in the dataset and record the checksum, and add it
    to this data structure.

    Attributes
    ----------
    train : str
        blake2s hash for the training split.
    test
        blake2s hash for the test split.
    validation
        blake2s hash for the validation split.
    predict
        blake2s hash for the predict split.
    """

    train: str | None = None
    test: str | None = None
    validation: str | None = None
    predict: str | None = None

    @staticmethod
    def string_is_hashlike(input_str: str) -> bool:
        """
        Simple method for checking if a string looks like a hash,
        which is just a string of lowercase alphanumerals.

        Parameters
        ----------
        input_str : str
            String to check if it looks like a hash.

        Returns
        -------
        bool
            True if the string appears to be a hash, False
            otherwise.
        """
        lookup = re.compile(r"[0-9a-f]{64}")
        # returns None if there are no matches
        if lookup.match(input_str):
            return True
        return False

    @field_validator("*")
    @classmethod
    def check_hash_like(cls, value: str, info: ValidationInfo) -> str:
        is_string_like_hash = SplitHashSchema.string_is_hashlike(value)
        if not is_string_like_hash:
            raise ValueError(
                f"Entry for {info.field_name} does not appear to be a hash."
            )
        return value

    @model_validator(mode="after")
    def check_not_all_none(self) -> Self:
        if not any(
            [getattr(self, key) for key in ["train", "test", "validation", "predict"]]
        ):
            raise RuntimeError("No splits were defined.")
        return self


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

    @field_validator("std")
    @classmethod
    def std_must_be_positive(cls, value: float) -> float:
        if value < 0.0:
            raise ValidationError("Standard deviation cannot be negative.")
        return value

    def to_normalizer(self) -> Normalizer:
        """
        Create ``Normalizer`` object for compatiability with training
        pipelines.

        TODO refactor the ``Normalizer`` class to be needed, and just
        use this class directly.

        Returns
        -------
        Normalizer
            Normalizer object used for computation.
        """
        return Normalizer(mean=self.mean, std=self.std)


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
    kwargs: dict[str, Any] = Field(default_factory=dict)

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

    def to_transform(self) -> PeriodicPropertiesTransform:
        """
        Generates the transform responsible for graph edge computation
        based on the schema.

        Returns
        -------
        PeriodicPropertiesTransform
            Instance of the periodic properties transform with
            schema settings mapped.
        """
        if self.algorithm in ["pymatgen", "ase"]:
            possible_kwargs = get_all_args(PeriodicPropertiesTransform)
            valid_kwargs = {
                key: value
                for key, value in self.kwargs.items()
                if key in possible_kwargs
            }
            valid_kwargs.setdefault("adaptive", True)
            return PeriodicPropertiesTransform(
                cutoff_radius=self.cutoff_radius, backend=self.algorithm, **valid_kwargs
            )
        else:
            raise NotImplementedError(
                "Custom backend for neighborhood algorithm not supported yet."
            )


class DatasetSchema(BaseModel):
    """
    A schema for defining a collection of data samples.

    This schema is to accompany all serialized datasets, which
    simultaneously documents the data **and** improves its
    reproducibility by defining

    Attributes
    ----------
    name : str
        Name of the dataset.
    creation : datetime
        An immutable ``datetime`` for when the dataset was
        created.
    target_keys : list[str]
        List of keys that are expected to be treated as target
        labels. This is used by the data pipeline to specifically
        load and designate as targets.
    split_blake2s : SplitHashSchema
        Schema representing blake2s checksums for each dataset split.
    modified : datetime, optional
        Datetime object for recording when the dataset was last
        modified.
    description : str, optional
        An optional, but highly recommended string for describing the
        nature and origins of this dataset. There is no limit to how
        long this description is, but ideally should be readable by
        humans and whatever is not obvious (such as what target key
        represents what property) should be included here.
    graph_schema : GraphWiringSchema, optional
        A schema that defines how the dataset is intended to build
        edges. This defines dictates how edges are created at runtime.
    normalization : dict[str, NormalizationSchema], optional
        Defines a collection of normalization mean/std for targets.
        If not None, this schema will validate against ``target_keys``
        and raise an error if there are keys in ``normalization`` that
        do not match ``target_keys``.
    node_stats : NormalizationSchema, optional
        Mean/std values for the nodes per data sample.
    edge_stats : NormalizationSchema, optional
        Mean/std values for the number of edges per data sample.
    """

    name: DatasetEnum
    creation: datetime
    target_keys: list[str]
    split_blake2s: SplitHashSchema
    dataset_type: str | DataSampleEnum | list[DataSampleEnum | str]
    modified: datetime | None = None
    description: str | None = None
    graph_schema: GraphWiringSchema | None = None
    normalization: dict[str, NormalizationSchema] | None = None
    node_stats: NormalizationSchema | None = None
    edge_stats: NormalizationSchema | None = None

    @classmethod
    def from_json(cls, json_path: PathLike) -> Self:
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

    @field_validator("dataset_type")
    @classmethod
    def cast_dataset_type(
        cls, value: str | DataSampleEnum | list[DataSampleEnum | str]
    ) -> list[DataSampleEnum]:
        """
        Validate and cast string values into enums.

        Returns
        -------
        list[DataSampleEnum]
            Regardless of the number of types specified, return
            a list of enum(s).
        """
        if isinstance(value, str):
            value = DataSampleEnum(value)
            assert value
        if isinstance(value, list):
            temp = []
            for subvalue in value:
                if isinstance(subvalue, str):
                    subvalue = DataSampleEnum(subvalue)
                    assert subvalue
                temp.append(subvalue)
            value = temp
        else:
            value = [value]
        return value

    @model_validator(mode="after")
    def check_target_normalization(self) -> Self:
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
    datatype: DataSampleEnum
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
