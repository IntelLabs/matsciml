from __future__ import annotations

from typing import Literal

import numpy as np
import torch
from pymatgen.core import Lattice, Structure
from loguru import logger
from ase.cell import Cell

from matsciml.common.types import DataDict
from matsciml.datasets.transforms.base import AbstractDataTransform
from matsciml.datasets.utils import (
    calculate_periodic_shifts,
    calculate_ase_periodic_shifts,
    make_pymatgen_periodic_structure,
)

__all__ = ["PeriodicPropertiesTransform"]


class PeriodicPropertiesTransform(AbstractDataTransform):
    def __init__(
        self,
        cutoff_radius: float,
        adaptive_cutoff: bool = False,
        backend: Literal["pymatgen", "ase"] = "pymatgen",
        max_neighbors: int = 1000,
        allow_self_loops: bool = False,
        convert_to_unit_cell: bool = False,
        is_cartesian: bool | None = None,
        is_undirected: bool = False,
    ) -> None:
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

        Parameters
        ----------
        cutoff_radius : float
            Cutoff radius to use to truncate the neighbor list calculation.
        adaptive_cutoff : bool, default False
            If set to ``True``, will allow ``cutoff_radius`` to grow up to
            30 angstroms if there are any disconnected subgraphs present.
            This is to allow distant nodes to be captured in some structures
            only as needed, keeping the computational requirements low for
            other samples within a dataset.
        backend : Literal['pymatgen', 'ase'], default 'pymatgen'
            Which algorithm to use for the neighbor list calculation. Nominally
            settings can be mapped to have the two produce equivalent results.
            'pymatgen' is kept as the default, but at some point 'ase' will
            become the default option. See the hosted documentation 'Best practices'
            page for details.
        max_neighbors : int, default 1000
            Forcibly truncate the number of edges at any given node. Internally,
            a counter is used to track the number of destination nodes when
            looping over a node's neighbor list; when the counter exceeds this
            value we immediately stop counting neighbors for the current node.
        allow_self_loops : bool, default False
            If ``True``, the edges will include self-interactions within the
            original unit cell. If set to ``False``, these self-loops are
            purged before returning edges.
        convert_to_unit_cell : bool, default False
            This argument is specific to ``pymatgen``, which is passed to the
            ``to_unit_cell`` argument during the ``Structure`` construction step.
        is_cartesian : bool | None, default None
            If set to ``None``, we will try and determine if the structure has
            fractional coordinates as input or not. If a boolean is provided,
            this is passed into the ``pymatgen.Structure`` construction step.
            This is specific to ``pymatgen``, and is not used by ``ase``.
        """
        super().__init__()
        self.cutoff_radius = cutoff_radius
        self.adaptive_cutoff = adaptive_cutoff
        self.backend = backend
        self.max_neighbors = max_neighbors
        self.allow_self_loops = allow_self_loops
        if is_cartesian is not None and backend == "ase":
            logger.warning(
                "`is_cartesian` passed but using `ase` backend; option will not affect anything."
            )
        self.is_cartesian = is_cartesian
        self.convert_to_unit_cell = convert_to_unit_cell
        self.is_undirected = is_undirected

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
                    structure,
                    self.cutoff_radius,
                    self.adaptive_cutoff,
                    max_neighbors=self.max_neighbors,
                    is_undirected=self.is_undirected,
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
            cell = Cell.new([*abc, *angles])
            # We need cell in data for ase backend.
            data["cell"] = cell.array
            lattice = data["cell"]

        if self.backend == "pymatgen":
            structure = make_pymatgen_periodic_structure(
                data["atomic_numbers"],
                data["pos"],
                lattice=lattice,
                convert_to_unit_cell=self.convert_to_unit_cell,
                is_cartesian=self.is_cartesian,
            )
            graph_props = calculate_periodic_shifts(
                structure,
                self.cutoff_radius,
                self.adaptive_cutoff,
                self.max_neighbors,
                self.is_undirected,
            )
        elif self.backend == "ase":
            graph_props = calculate_ase_periodic_shifts(
                data,
                self.cutoff_radius,
                self.adaptive_cutoff,
                self.max_neighbors,
                self.is_undirected,
            )
        else:
            raise RuntimeError(f"Requested backend f{self.backend} not available.")
        data.update(graph_props)
        if not self.allow_self_loops:
            # this looks for src and dst nodes that are the same, i.e. self-loops
            loop_mask = data["src_nodes"] == data["dst_nodes"]
            # only mask out self-loops within the same image
            images = data["images"]
            image_mask = (images[:, 0] == 0) & (images[:, 1] == 0) & (images[:, 2] == 0)
            # we negate the mask because we want to *exclude* what we've found
            mask = ~torch.logical_and(loop_mask, image_mask)
            # apply mask to each of the tensors that depend on edges
            for key in ["src_nodes", "dst_nodes", "images", "unit_offsets", "offsets"]:
                data[key] = data[key][mask]
        return data
