from __future__ import annotations

import torch
from pymatgen.core import Lattice

from matsciml.common.types import DataDict
from matsciml.datasets.transforms.base import AbstractDataTransform
from matsciml.datasets.utils import (
    calculate_periodic_shifts,
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
    """

    def __init__(self, cutoff_radius: float) -> None:
        super().__init__()
        self.cutoff_radius = cutoff_radius

    def __call__(self, data: DataDict) -> DataDict:
        for key in ["atomic_numbers", "pos"]:
            assert key in data, f"{key} missing from data sample!"
        if "cell" in data:
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
                raise KeyError(
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
            lattice = Lattice.from_parameters(*abc, *angles)
        structure = make_pymatgen_periodic_structure(
            data["atomic_numbers"],
            data["pos"],
            lattice=lattice,
        )
        graph_props = calculate_periodic_shifts(structure, self.cutoff_radius)
        data.update(graph_props)
        return data
