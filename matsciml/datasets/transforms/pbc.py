from __future__ import annotations

import numpy as np
import torch
from pymatgen.core import Lattice, Structure

from matsciml.common.types import DataDict
from matsciml.datasets.transforms.base import AbstractDataTransform
from matsciml.datasets.utils import (
    calculate_periodic_shifts,
    calculate_ase_periodic_shifts,
    make_pymatgen_periodic_structure,
)

__all__ = ["PeriodicPropertiesTransform"]


class PeriodicPropertiesTransform(AbstractDataTransform):
    """
    Rewires an already present graph to include periodic boundary conditions.

    Since graphs are normally bounded within a unit cell, they may not capture
    the necessary dependencies for atoms connected to neighboring cells. This
    transform will compute the unit cell, tile it, and then rewire the graph
    edges such that it can capture connectivity given a radial cutoff given
    in Angstroms.

    Cut off radius is specified in Angstroms. An additional flag, ``adaptive_cutoff``,
    allows the cut off value to grow up to 100 angstroms in order to find neighbors.
    This allows larger (typically unstable) structures to be modeled without applying
    a large cut off for the entire dataset.
    """

    def __init__(
        self,
        cutoff_radius: float,
        adaptive_cutoff: bool = False,
        backend: str = "pymatgen",
    ) -> None:
        super().__init__()
        self.cutoff_radius = cutoff_radius
        self.adaptive_cutoff = adaptive_cutoff
        self.backend = backend

    def __call__(self, data: DataDict) -> DataDict:
        """
        Given a data sample, generate graph edges with periodic boundary conditions
        as specified by ``cutoff_radius`` and ``adaptive_cutoff``.

        This function has several nested conditions, depending on the availability
        of data pertaining to periodic structures. First and foremost, if there
        is a serialized ``pymatgen.core.Structure`` object, we will take that
        directly and use it to compute the periodic shifts as to minimize ambiguituies.
        If there isn't one available, we then check for the presence of a ``cell``
        or lattice matrix, from which we use to create a ``Lattice`` object that
        is *then* used to create a ``pymatgen.core.Structure``. If a ``cell`` isn't
        available, the final check is to look for keys related to lattice parameters,
        and use those instead.

        Parameters
        ----------
        data : DataDict
            Data sample retrieved from a dataset.

        Returns
        -------
        DataDict : DataDict
            Data sample, now with updated key/values based on periodic
            properties. See ``calculate_periodic_shifts`` for the additional
            keys.

        Raises
        ------
        RuntimeError:
            If the final check for lattice parameters fails, there is nothing
            we can base the periodic boundary calculation off of.
        """
        for key in ["atomic_numbers", "pos"]:
            assert key in data, f"{key} missing from data sample!"
        # if we have a pymatgen structure serialized already use it directly
        if "structure" in data:
            structure = data["structure"]
            if isinstance(structure, Structure):
                graph_props = calculate_periodic_shifts(
                    structure, self.cutoff_radius, self.adaptive_cutoff
                )
                data.update(graph_props)
                return data
        # continue this branch if the structure doesn't qualify
        if "cell" in data:
            assert isinstance(
                data["cell"], (torch.Tensor, np.ndarray)
            ), "Lattice matrix is not array-like."
            # squeeze is used to make sure we remove empty dims
            lattice = Lattice(data["cell"].squeeze())
        else:
            # if we don't have a cell already, we go through the
            # whole process
            if "lattice_features" in data:
                lattice_key = "lattice_features"
            elif "lattice_params" in data:
                lattice_key = "lattice_params"
            else:
                raise RuntimeError(
                    "Data sample is missing lattice parameters. "
                    "Ensure `lattice_features` or `lattice_params` is available"
                    " in the data.",
                )
            lattice_params: torch.Tensor = data[lattice_key]
            if isinstance(lattice_params, dict):
                lattice_params = lattice_params["lattice_params"]
            abc, angles = lattice_params[:3], lattice_params[3:]
            angles = torch.FloatTensor(
                tuple(angle * (180.0 / torch.pi) for angle in angles),
            )
            lattice = Lattice.from_parameters(*abc, *angles, vesta=True)
            # We need cell in data for ase backend.
            data["cell"] = torch.tensor(lattice.matrix).unsqueeze(0).float()

        structure = make_pymatgen_periodic_structure(
            data["atomic_numbers"],
            data["pos"],
            lattice=lattice,
        )
        if self.backend == "pymatgen":
            graph_props = calculate_periodic_shifts(
                structure, self.cutoff_radius, self.adaptive_cutoff
            )
        elif self.backend == "ase":
            graph_props = calculate_ase_periodic_shifts(
                data, self.cutoff_radius, self.adaptive_cutoff
            )
        else:
            raise RuntimeError(f"Requested backend f{self.backend} not available.")
        data.update(graph_props)
        return data
