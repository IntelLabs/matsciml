from __future__ import annotations

import torch
from einops import rearrange
from pymatgen.core import Lattice

from matsciml.common.packages import package_registry
from matsciml.common.types import DataDict
from matsciml.datasets.transforms.base import AbstractDataTransform

if any([pkg in package_registry for pkg in ["pyg", "dgl"]]):
    _has_graphs = True
else:
    _has_graphs = False


if _has_graphs:

    class PeriodicGraphTransform(AbstractDataTransform):
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
            assert (
                "graph" in data
            ), "Missing graph in data sample; please prepend a graph transform."
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
            abc, angles = lattice_params[:3], lattice_params[3:]
            angles = torch.FloatTensor(
                tuple(angle * (180.0 / torch.pi) for angle in angles),
            )
            lattice_obj = Lattice.from_parameters(*abc, *angles)
            lattice_matrix = torch.Tensor(lattice_obj.matrix)
            # add dimension for batching
            data["cell"] = rearrange(lattice_matrix, "h w -> () h w")
            return data
