from __future__ import annotations

from importlib import import_module
from enum import Enum
from datetime import datetime
from typing import Literal, Any, Self
from os import PathLike
from pathlib import Path
import re

from ase import Atoms
from ase.geometry import cell_to_cellpar, cellpar_to_cell
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    create_model,
    field_validator,
    model_validator,
    ValidationInfo,
)
from numpydantic import NDArray, Shape
from loguru import logger
import numpy as np
import torch

from matsciml.common.packages import package_registry
from matsciml.common.inspection import get_all_args
from matsciml.common.types import Embeddings, ModelOutput
from matsciml.modules.normalizer import Normalizer
from matsciml.datasets.transforms import PeriodicPropertiesTransform
from matsciml.datasets.utils import cart_frac_conversion
from matsciml.datasets import validators as v

"""This module defines schemas pertaining to data, using ``pydantic`` models
to help with validation and (de)serialization.

The driving principle behind this is to try and define a standardized data
format that is also relatively low maintenance. The ``DatasetSchema`` is
used to fully qualify a dataset, including file hashing, a list of expected
targets, and so on. The ``DataSampleSchema`` provides an "on-the-rails"
experience for both developers and users by defining a consistent set of
attribute names that should be
"""

__all__ = [
    "DataSampleSchema",
    "DataSampleEnum",
    "DatasetSchema",
    "SplitHashSchema",
    "PeriodicBoundarySchema",
    "NormalizationSchema",
    "GraphWiringSchema",
    "TargetSchema",
    "collate_samples_into_batch_schema",
]

# ruff: noqa: F722


class MatsciMLSchema(BaseModel):
    """
    Implements a base class with (de)serialization methods
    for saving and loading JSON files.
    """

    @classmethod
    def from_json_file(cls, json_path: PathLike) -> Self:
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

    def to_json_file(self, json_path: PathLike) -> None:
        """
        Write out the schema to a JSON file.

        Parameters
        ----------
        json_path : PathLike
            Filepath to save the data to. If unspecified, the
            suffix ``.json`` will be used.
        """
        if not isinstance(json_path, Path):
            json_path = Path(json_path)
        with open(json_path.with_suffix(".json"), "w+") as write_file:
            write_file.write(self.model_dump_json(round_trip=True, indent=2))


class DataSampleEnum(str, Enum):
    """
    An Enum for categorizing data samples, which implicitly
    informs us what the sample is intended to be used for
    by virtue of what data is available, as well as how data
    samples may relate to one another. An example would be
    ``OptimizationCycle``, which implies the dataset should
    contain multiple samples per structure of atomic forces.

    These tend to map more directly to computational chemistry
    workflows, although naturally some types of calculations
    will have overlap between them (e.g. an excited state geometry
    optimization). In those cases, the recommendation would be
    to select the intended use case - i.e. ``OptimizationCycle``
    is the preferred enum for this example as it infers the
    presence of atomic forces.

    Attributes
    ----------
    scf : str
        Describes data pertaining to a single SCF cycle, which
        comprises energy values, population analyses, orbital
        coefficients, spin states, convergence properties etc.
    opt_trajectory : str
        Describes data comprising a single optimization or relaxation
        step, which includes atomic forces, (partial) Hessians,
        and geometry convergence metrics.
    e_property : str
        Describes a specific electronic property calculation. This can range
        from multipole moments, to polarization, etc. Choose this
        category if your intention is to provide properties, even if
        certain electronic properties come for 'free' with SCF calculations.
    n_property : str
        Describes a specific nuclear property calculation, such as nuclear
        multipole moments (e.g. nitrogen quadrupole), magnetic moments, etc.
    states : str
        Describes an excited state calculation that does not involve geometry
        optimizations. This may refer to oscillator strengths/transition
        moments.
    """

    scf = "SCFCycle"
    opt_trajectory = "OptimizationCycle"
    e_property = "ElectronicPropertyCalculation"
    n_property = "NuclearPropertyCalculation"
    states = "ExcitedStateCalculation"


class SplitHashSchema(MatsciMLSchema):
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
        if value is not None:
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


class NormalizationSchema(MatsciMLSchema):
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


class GraphWiringSchema(MatsciMLSchema):
    """
    Provides a specification for tracking how graphs within
    a dataset are wired. Primarily, a ``cutoff_radius`` is
    specified to artificially truncate a neighborhood, and
    a package or algorithm is used to compute this neighborhood
    function.

    The validation of this schema includes checking to ensure
    that a specific package used for performing neighborhood
    calculations matches the recorded version, to ensure that
    there are no unexpected changes to the algorithm used.

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
    adaptive_cutoff: bool
    algo_version: str | None = None
    algo_hash_path: str | None = None
    algo_hash: str | None = None
    max_neighbors: int = -1
    kwargs: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_transform(
        cls, pbc_transform: PeriodicPropertiesTransform, allow_mismatch: bool
    ) -> Self:
        package = pbc_transform.backend
        version = cls._check_package_version(package)
        return cls(
            cutoff_radius=pbc_transform.cutoff_radius,
            algorithm=pbc_transform.backend,
            allow_mismatch=allow_mismatch,
            algo_version=version,
            adaptive_cutoff=pbc_transform.adaptive_cutoff,
            max_neighbors=pbc_transform.max_neighbors,
            kwargs={
                "is_cartesian": pbc_transform.is_cartesian,
                "allow_self_loops": pbc_transform.allow_self_loops,
                "convert_to_unit_cell": pbc_transform.convert_to_unit_cell,
            },
        )

    @staticmethod
    def _check_package_version(backend: str) -> str | None:
        """Simple function for checking the version of pymatgen/ase"""
        if backend == "pymatgen":
            pmg = import_module("pymatgen.core")
            actual_version = pmg.__version__
        elif backend == "ase":
            ase = import_module("ase")
            actual_version = ase.__version__
        else:
            logger.warning("Periodic backend unsupported and cannot check version.")
            actual_version = None
        return actual_version

    @model_validator(mode="after")
    def check_algo_version(self):
        if not self.algo_version and not self.algo_hash:
            raise RuntimeError("At least one form of algorithm versioning is required.")
        if self.algo_version:
            actual_version = self._check_package_version(self.algorithm)
            if actual_version is None:
                return self
            # throw validation error only if we don't allow mismatches
            if self.algo_version != actual_version and not self.allow_mismatch:
                raise RuntimeError(
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
                raise RuntimeError(
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
            return PeriodicPropertiesTransform(
                cutoff_radius=self.cutoff_radius, backend=self.algorithm, **valid_kwargs
            )
        else:
            raise NotImplementedError(
                "Custom backend for neighborhood algorithm not supported yet."
            )


class TargetSchema(MatsciMLSchema):
    """
    Schema that specifies a target label or property.

    The intention is to provide sufficient fields to make targets
    in a dataset fully documented to leave zero ambiguities.

    Attributes
    ----------
    name : str
        String name of the target. This will be used to look up
        targets throughout the pipeline.
    shape : str
        Designated shape of the target. Use '*' to specify variable
        dimensions, and integers for fixed dimensions separated
        by commas. As an example, '*' could designate a number of
        node features (since the number of nodes is variable), and
        '*, 3' could represent a vector property also over nodes.
    description : str
        Long text description of what this target is and how it
        was calculated.
    units : str, optional
        Expected units of this property. This is more for documentation
        for now, but in the future it may be helpful to do unit
        conversions with this field.
    """

    name: str
    shape: str
    description: str
    units: str | None = None

    @model_validator(mode="after")
    def check_shape_str(self) -> Self:
        """This checks to make sure that the shape specification is valid for ``Shape``."""
        invalid_regex = re.compile(r"[^\d\,\*\s]+")
        if invalid_regex.search(self.shape):
            raise ValueError(
                f"Target shape should be specified with digits, commas, and wildcard only. Got {self.shape}"
            )
        return self


class DatasetSchema(MatsciMLSchema):
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
    targets : list[TargetSchema]
        A list of ``TargetSchema`` objects or dictionaries that satisfy
        the schema. This is used simultaneously for documentation as
        well as for data loading to look specifically for these keys.
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
    seed : int, optional
        Random seed used to generate the splits. This is kept optional
        in the case where splits are not randomly generated.
    """

    name: str
    creation: datetime
    targets: list[TargetSchema]
    split_blake2s: SplitHashSchema
    dataset_type: DataSampleEnum | list[DataSampleEnum]
    modified: datetime | None = None
    description: str | None = None
    graph_schema: GraphWiringSchema | None = None
    normalization: dict[str, NormalizationSchema] | None = None
    node_stats: NormalizationSchema | None = None
    edge_stats: NormalizationSchema | None = None
    seed: int | None = None

    @classmethod
    def from_json_file(cls, json_path: PathLike) -> Self:
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
        """Cross-check target normalization specification with defined targets."""
        if self.normalization is not None:
            # first check every key is available as targets
            target_keys = set([target.name for target in self.targets])
            norm_keys = set(self.normalization.keys())
            # check to see if we have unexpected norm keys
            diff = norm_keys - target_keys
            if len(diff) > 0:
                raise ValidationError(f"Unexpected keys in normalization: {diff}")
        return self


class DataSampleSchema(MatsciMLSchema):
    """
    Defines a schema for a single data sample.

    Includes fields for the most commonly used properties, particularly
    for interatomic potentials, and includes additional ones that help
    fully specify the state of a atomic structure/material, such as
    isotopic masses, charges, and electronic states.

    This schema uses ``numpydantic`` for type and shape hinting;
    it does not enforce what provides the array (e.g. ``torch``
    or ``numpy``), and when dumping to JSON will ensure that it
    is serializable (i.e. converts it to a list first). We also implement
    a consistency check after model creation to validate that per-atom
    fields have the right number of atoms.

    Parameters
    ----------
    index : int
        Integer counter for the sample within the full dataset.
        This value is used to uniquely identify the sample within the
        dataset, and is helpful during debugging.
    num_atoms : int
        Specifies the number of atoms to be expected of this data
        sample. Recording this explicitly makes it accessible,
        instead of relying on determination after batching, etc.
        with tricks like ``graph.ptr``.
    cart_coords : NDArray[Shape['*, 3'], float]
        A variable length array of 3D vectors of floating point numbers
        corresponding to the cartesian coordinates of the structure.
    atomic_numbers : NDArray[Shape['*'], int]
        A variable length array of integers corresponding to the
        atomic numbers of each species.
    pbc : PeriodicBoundarySchema
        A schema that specifies which axes are periodic. Can also
        pass a dictionary of coordinate/``bool`` values instead of
        constructing the ``PeriodicBoundarySchema`` ahead of time.
    datatype : DataSampleEnum
        Categorizes the data sample according to types defined in
        the ``DataSampleEnum``. This is mainly for documentation,
        but allows users/developers to expect certain data fields
        to be populated.
    alpha_electron_spins : NDArray[Shape['*'], float], optional
        Specifies the alpha spin value per-atom as a variable length
        array of floating point values. Assumes unrestricted/open
        shell species; alternatively, specify the same number in
        ``beta_electron_spins``.
    beta_electron_spins : NDArray[Shape['*'], float], optional
        Specifies the beta spin value per-atom as a variable length
        array of floating point values. Assumes unrestricted/open
        shell species; alternatively, specify the same number in
        ``alpha_electron_spins``.
    nuclear_spins : NDArray[Shape['*'], float], optional
        Specifies the nuclear spin value per-atom as a variable
        length array of floating point values.
    isotopic_masses : NDArray[Shape['*'], float], optional
        Specifies isotopic masses for each atom as a variable
        length array of floating point values.
    atomic_charges : NDArray[Shape['*'], float], optional
        Specifies some characterization of charge for each atom
        as a variable length array of floating point values.
    atomic_energies : NDArray[Shape['*'], float], optional
        Ascribes energy values - i.e. contributions from each atom -
        to each atom in the sample as a variable length array of
        floating point values.
    atomic_labels : NDArray[Shape['*'], int], optional
        Indices to 'tag' atoms - useful for classification tasks
        and masking. Specified as a variable length array of integers.
    total_energy : float, optional
        Total energy of the system by whatever definition. If there
        are multiple types of total energy values, we recommend writing
        the most primitive type (e.g. total electronic energy) available,
        and add others (e.g. corrections, etc.) to ``extra``.
    forces : NDArray[Shape['*, 3'], float], optional
        Specifies atomic forces on each atom as a variable length
        array of 3D vectors with floating point values.
    stresses : NDArray[Shape['*, 3, 3'], float], optional
        Specifies a stress tensor per atom as a variable length
        array of 3x3 matrices of floating point values.
    lattice_parameters : NDArray[Shape['6'], float], optional
        Specifies a vector of lattice parameters in order of
        ``a,b,c,alpha,beta,gamma``. Assumes angles ``alpha,beta,gamma``
        are in degrees.
    lattice_matrix : NDArray[Shape['3, 3'], float], optional
        Specifies the fully specified lattice matrix as a 3x3 matrix
        of floating point values. If the choice is between this field
        or ``lattice_parameters``, populate this field but ideally both
        as the matrix generated from parameters may not be unique.
    edge_index : NDArray[Shape['2, *'], int], optional
        Indices to indicate edges between atoms as a variable length
        array of 2D vectors. The variable length in this case corresponds
        to the number of **edges**, not atoms/nodes.
    charge : float, optional
        Some characterization of charge for the whole system as a floating
        point value.
    multiplicity : float, optional
        Electronic multiplicity of the system as a floating point value.
        While not explicitly checked, this couples with ``electronic_state_index``
        to fully specify an electronic state. This value is defined as 2S+1,
        with S being the number of unpaired electrons (or the total electron spin
        angular momentum).
    electronic_state_index : int, default 0
        Specifies the electronic state, with zero (default) being the
        electronic ground state for a given multiplicity. The index
        should be ordered by energy, i.e. the first singlet excited state
        would be given by a value of 1, and a multiplicity of 0.
    images : NDArray[Shape['*, 3'], int], optional
        Variable length array of 3D vectors of integers that index
        periodic images (i.e. neighboring unit cells). The length
        is expected to match that of ``edge_index``.
    offsets : NDArray[Shape['*, 3'], float], optional
        Variable length array of 3D vectors of floating points that
        can be used to shift the point of reference from the origin
        unit cell to the corresponding periodic image. The length
        should be the same as ``edge_index``/``images``.
    unit_offsets : NDArray[Shape['*, 3'], float], optional
        Builds on top of ``offsets``, including the difference in
        positions between two atoms in fractional coordinates.
        Expects the length to be the same as ``edge_index``/``images``
        and ``offsets``.
    graph : Any, optional
        This field is not intended to be serialized, but is included
        to allow the field to be populated during runtime as a way
        to store an arbitrary graph object. We do not want to serialize
        the graph object directly, as reloading with version mismatches
        can be made impossible with breaking API changes regardless of
        the framework. Instead, opt to save ``edge_index``.
    extras : dict[str, Any], optional
        Provides a vehicle for out-of-spec data to be transported in
        this schema. This is useful if the property you wish to save
        does not fit under any of the currently defined fields, but
        is not recommended as it bypasses any of the type and shape
        validations that ``pydantic``/``numpydantic`` provides.
    transform_store : dict[str, Any], optional
        Dictionary storage for transform results. This is a way to organize
        products of transforms, e.g. instead of overwriting properties.
    """

    index: int
    num_atoms: int
    cart_coords: v.CoordTensor
    atomic_numbers: v.Long1DTensor
    pbc: PeriodicBoundarySchema
    datatype: DataSampleEnum
    alpha_electron_spins: v.Float1DTensor | None = None
    beta_electron_spins: v.Float1DTensor | None = None
    nuclear_spins: v.Float1DTensor | None = None
    isotopic_masses: v.Float1DTensor | None = None
    atomic_charges: v.Float1DTensor | None = None
    atomic_energies: v.Float1DTensor | None = None
    atomic_labels: v.Long1DTensor | None = None
    total_energy: float | None = None
    forces: v.CoordTensor
    stresses: v.StressTensor
    lattice_parameters: v.LatticeParameters | None = None
    lattice_matrix: v.LatticeTensor
    edge_index: v.EdgeTensor
    frac_coords: v.CoordTensor | None = None
    charge: float | None = None  # overall system charge
    multiplicity: float | None = None  # overall system multiplicity
    electronic_state_index: int = 0
    images: v.CoordTensor | None = None
    offsets: v.CoordTensor | None = None
    unit_offsets: v.CoordTensor | None = None
    graph: Any = None
    extras: dict[str, Any] | None = None
    transform_store: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(arbitrary_types_allowed=True, use_enum_values=True)

    def __getattr__(self, name: str) -> Any | None:
        """Overrides the behavior of `getattr` to also look in `extras` if available"""
        if name in self.__dir__():
            return self.__dict__[name]
        if self.extras is not None and name in self.extras:
            return self.extras[name]
        return None

    def _exception_wrapper(self, exception: Exception):
        """
        Re-raises an exception that uses this class, and chains the sample index.

        This is to make debugging more informative, as it allows
        arbitrary exceptions to be raised while also informing us
        which sample specifically is causing issues.

        Parameters
        ----------
        exception : Exception
            Any possible ``Exception``. The type of exception is
            used to re-raise the exception including the sample index.

        Raises
        ------
        exception_cls
            Raises the same exception as the input one, with
            an additional message.
        """
        exception_cls = exception.__class__
        raise exception_cls(
            f"Data schema validation failed at sample {self.index}."
        ) from exception

    @model_validator(mode="before")
    @classmethod
    def convert_lattice_and_parameters(cls, values: Any) -> Any:
        lattice_params = values.get("lattice_parameters", None)
        lattice_matrix = values.get("lattice_matrix", None)
        if lattice_params is None and lattice_matrix is not None:
            lattice_params = cell_to_cellpar(lattice_matrix)
            values["lattice_parameters"] = lattice_params
        if lattice_params is not None and lattice_matrix is None:
            lattice_matrix = cellpar_to_cell(lattice_params)
            values["lattice_matrix"] = lattice_matrix
        return values

    @model_validator(mode="after")
    def coordinate_consistency(self) -> Self:
        """Sets fractional coordinates if parameters are available, and checks them"""
        if self.frac_coords is None and self.lattice_parameters is not None:
            self.frac_coords = cart_frac_conversion(
                self.cart_coords, *self.lattice_parameters, to_fractional=True
            )
        if isinstance(self.frac_coords, NDArray):
            if self.frac_coords.shape != self.cart_coords.shape:
                raise ValueError(
                    "Fractional coordinate dimensions do not match cartesians."
                )
            # round coordinate values so that -1e-6 is just zero and doesn't fail the test
            round_coords = np.round(self.frac_coords, decimals=2)
            if np.any(np.logical_or(round_coords > 1.01, round_coords < 0.0)):
                logger.warning(
                    f"Fractional coordinates are outside of [0, 1]: {round_coords}"
                )
        return self

    @model_validator(mode="after")
    def atom_count_consistency(self) -> Self:
        for key in [
            "atomic_numbers",
            "electron_spins",
            "nuclear_spins",
            "isotopic_masses",
            "atomic_charges",
            "atomic_energies",
            "atomic_labels",
        ]:
            value = getattr(self, key, None)
            if value is not None:
                if len(value) != self.num_atoms:
                    self._exception_wrapper(
                        ValueError(
                            f"Inconsistent number of elements for {key}; expected {self.num_atoms}, got {len(value)}."
                        )
                    )
        for key in ["forces", "stresses"]:
            value = getattr(self, key, None)
            if value is not None:
                if value.shape[0] != self.num_atoms:
                    self._exception_wrapper(
                        ValueError(
                            f"Inconsistent number of elements for node property {key}; expected {self.num_atoms}, got {value.shape[0]}."
                        )
                    )
        if self.edge_index is not None:
            for key in ["images", "offsets", "unit_offsets"]:
                value = getattr(self, key, None)
                if value is not None:
                    if value.shape[0] != self.edge_index:
                        self._exception_wrapper(
                            ValueError(
                                f"Inconsistent number of elements for edge property {key}."
                            )
                        )
        return self

    def __eq__(self, other: DataSampleSchema) -> bool:
        """Overrides the equivalence test, including array allclose comparisons"""
        assert isinstance(
            other, DataSampleSchema
        ), "Equal comparison can only be done against `DataSampleSchema`."
        self_dict = self.model_dump()
        other_dict = other.model_dump()
        for key in self_dict.keys():
            self_value = self_dict[key]
            other_value = other_dict[key]
            # skip None comparisons
            if self_value is None and other_value is None:
                continue
            try:
                if type(self_value) != type(other_value):
                    return False
                if isinstance(self_value, torch.Tensor):
                    check = torch.allclose(self_value, other_value)
                    if not check:
                        return False
                elif isinstance(self_value, np.ndarray):
                    check = np.allclose(self_value, other_value)
                    if not check:
                        return False
                else:
                    # for everything else, str, int, float, etc. builtin types
                    if not self_value == other_value:
                        return False
            except Exception:
                # if at any point any exception is raised, they're
                # not equal
                logger.debug(f"Comparison failed on key {key}")
                return False
        return True

    @model_validator(mode="after")
    def check_edge_data(self) -> Self:
        """Ensure that if edge properties are consistent with number of edges."""
        if self.edge_index is not None:
            num_edges = self.edge_index.shape[1]
            for key in ["images", "offsets", "unit_offsets"]:
                value = getattr(self, key)
                if value is not None:
                    if value.shape[0] != num_edges:
                        self._exception_wrapper(
                            ValueError(
                                f"Mismatch in edge property {key}. "
                                "Expected the first dimension to match the number of edges."
                            )
                        )
        return self

    def to_ase_atoms(self) -> Atoms:
        """
        Provides a simple conversion to an ``ase.Atoms`` object.

        This method does not strictly check that outputs are mapped
        correctly, but at least maps the fields in the schema to
        intended attributes in the ``Atoms`` class.

        Returns
        -------
        Atoms
            Instance of an ``Atoms`` object constructed with
            the current data sample.
        """
        pbc = [value for value in self.pbc.model_dump().values()]
        return Atoms(
            positions=self.cart_coords,
            cell=self.lattice_matrix,
            numbers=self.atomic_numbers,
            tags=self.atomic_labels,
            charges=self.atomic_charges,
            masses=self.isotopic_masses,
            pbc=pbc,
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
                self._exception_wrapper(
                    TypeError(f"Unexpected graph type: {type(self.graph)}")
                )


def _concatenate_data_list(all_data: list[Any]) -> list[Any] | torch.Tensor:
    """
    Concatenates an arbitrary list of data where possible.

    For array-types (NumPy, PyTorch), we first convert all
    of the sample data into tensors, followed by concatenation
    along the first dimension (nodes or edges).

    For scalar-types, we return a 1D tensor.

    For all other types, we return the inputs unmodified.

    Parameters
    ----------
    all_data : list[Any]
        List of data to try and concatenate.

    Returns
    -------
    list[Any] | torch.Tensor
        If the concatenation was successful, returns a tensor.
        Otherwise, returns the unmodified input.
    """
    sample = all_data[0]
    if isinstance(sample, (np.ndarray, torch.Tensor)):
        # homogenize all samples into tensors
        all_data = [torch.Tensor(s) for s in all_data]
        return torch.concat(all_data)
    if isinstance(sample, (float, int)):
        output = torch.Tensor(all_data)
        if isinstance(sample, int):
            output = output.long()
        return output
    else:
        # leave the data as a list
        return all_data


def collate_samples_into_batch_schema(samples: list[DataSampleSchema]) -> object:
    """
    Function to collate a list of ``DataSampleSchema`` into a dynamically
    generated ``BatchSchema``.

    The additional logic in this function is to handle graphs, copying
    references to their respective data, and calling the respective framework
    batching functions.

    The purpose of on-the-fly schema generation is to do some degree of
    validation, but primarily provide regular structure for use in the
    task pipeline side of things. Given that the schema should be serializable,
    it may also make debugging more streamlined.

    Parameters
    ----------
    samples : list[DataSampleSchema]
        List of data samples that have been pre-validated.

    Returns
    -------
    object
        Instance of a ``BatchSchema`` object. This is not explicitly annotated
        since the model/class is defined dynamically based off incoming data.
    """
    ref_schema = samples[0].model_json_schema()
    # initial keys are going to hold the main structure of the schema
    schema_to_generate = {
        "num_atoms": (NDArray[Shape["*"], int] | torch.LongTensor, ...),
        "batch_size": (int, ...),
        "graph": (Any | None, None),
        "num_edges": (NDArray[Shape["*"], int] | torch.LongTensor | None, None),
        "embeddings": (Embeddings | None, None),
        "outputs": (ModelOutput | None, None),
    }
    collected_data = {}  # holds all the data to unpack into the generated schema
    # check to see if graphs are present
    if samples[0].graph is not None:
        graph_sample = samples[0].graph
        if "pyg" in package_registry:
            from torch_geometric.data import Batch, Data

            if isinstance(graph_sample, Data):
                batched_graph = Batch.from_data_list(
                    [sample.graph for sample in samples]
                )
                graph_type = Batch
                for key in batched_graph.keys():
                    data = getattr(batched_graph, key)
                    schema_to_generate[key] = (type(data), ...)
                    collected_data[key] = data
                collected_data["num_edges"] = _concatenate_data_list(
                    [sample.graph.edge_index.size(-1) for sample in samples]
                ).long()
        else:
            from dgl import DGLGraph, batch

            if isinstance(graph_sample, DGLGraph):
                batched_graph = batch([sample.graph for sample in samples])
                graph_type = DGLGraph
                for key, data in batched_graph.ndata.items():
                    schema_to_generate[key] = (type(data), ...)
                    collected_data[key] = data
                for key, data in batched_graph.edata.items():
                    schema_to_generate[key] = (type(data), ...)
                    collected_data[key] = data
                collected_data["num_edges"] = _concatenate_data_list(
                    [sample.graph.batch_num_edges() for sample in samples]
                ).long()
        collected_data["num_atoms"] = _concatenate_data_list(
            [sample.num_atoms for sample in samples]
        ).long()
        collected_data["graph"] = batched_graph
        schema_to_generate["graph"] = (graph_type, ...)
    # for everything else that wasn't packed into the graph
    for key in ref_schema["required"]:
        if key not in schema_to_generate:
            schema_to_generate[key] = (Any, ...)
        if key not in collected_data:
            collected_data[key] = _concatenate_data_list(
                [getattr(sample, key) for sample in samples]
            )
    collected_data["batch_size"] = len(samples)
    # generate the schema, then create the model
    BatchSchema = create_model(
        "BatchSchema",
        **schema_to_generate,
        __config__=ConfigDict(arbitrary_types_allowed=True, use_enum_values=True),
    )
    return BatchSchema(**collected_data)
