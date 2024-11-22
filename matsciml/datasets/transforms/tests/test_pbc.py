from __future__ import annotations

from collections import Counter

import torch
import pytest
import numpy as np
from pymatgen.core import Structure, Lattice

from matsciml.datasets.transforms import PeriodicPropertiesTransform

"""
This module uses reference Materials project structures and tests
the edge calculation routines to ensure they at least work with
various parameters.

The key thing here is at least using feasible structures to perform
this check, rather than using randomly generated coordinates and
lattices, even if composing them isn't meaningful.
"""

hexa = Lattice.from_parameters(
    4.81, 4.809999999999999, 13.12, 90.0, 90.0, 120.0, vesta=True
)
cubic = Lattice.from_parameters(6.79, 6.79, 12.63, 90.0, 90.0, 90.0, vesta=True)

# mp-1143
alumina = Structure(
    hexa,
    species=["Al", "O"],
    coords=[[1 / 3, 2 / 3, 0.814571], [0.360521, 1 / 3, 0.583333]],
    coords_are_cartesian=False,
)
# mp-1267
nac = Structure(
    cubic,
    species=["Na", "C"],
    coords=[[0.688819, 3 / 4, 3 / 8], [0.065833, 0.565833, 0.0]],
    coords_are_cartesian=False,
)


@pytest.mark.parametrize(
    "coords",
    [
        alumina.cart_coords,
        nac.cart_coords,
    ],
)
@pytest.mark.parametrize(
    "cell",
    [
        hexa.matrix,
        cubic.matrix,
    ],
)
@pytest.mark.parametrize("self_loops", [True, False])
@pytest.mark.parametrize("backend", ["pymatgen", "ase"])
@pytest.mark.parametrize(
    "cutoff_radius", [6.0, 9.0, 15.0]
)  # TODO figure out why pmg fails on 3
def test_periodic_generation(
    coords: np.ndarray,
    cell: np.ndarray,
    self_loops: bool,
    backend: str,
    cutoff_radius: float,
):
    coords = torch.FloatTensor(coords)
    cell = torch.FloatTensor(cell)
    transform = PeriodicPropertiesTransform(
        cutoff_radius=cutoff_radius,
        adaptive_cutoff=False,
        backend=backend,
        max_neighbors=10,
        allow_self_loops=self_loops,
    )
    num_atoms = coords.size(0)
    atomic_numbers = torch.ones(num_atoms)
    packed_data = {"pos": coords, "cell": cell, "atomic_numbers": atomic_numbers}
    output = transform(packed_data)
    # check to make sure no source node has more than 10 neighbors
    src_nodes = output["src_nodes"].tolist()
    counts = Counter(src_nodes)
    for index, count in counts.items():
        if not self_loops:
            assert count < 10, print(f"Node {index} has too many counts. {src_nodes}")


def test_self_loop_condition():
    """Tests for whether the self-loops exclusion is behaving as intended"""
    coords = torch.FloatTensor(alumina.cart_coords)
    cell = torch.FloatTensor(alumina.lattice.matrix)
    num_atoms = coords.size(0)
    atomic_numbers = torch.ones(num_atoms)
    packed_data = {"pos": coords, "cell": cell, "atomic_numbers": atomic_numbers}
    no_loop_transform = PeriodicPropertiesTransform(
        cutoff_radius=6.0, backend="ase", allow_self_loops=False
    )
    no_loop_result = no_loop_transform(packed_data)
    # since it's no self loops this sum should be zero
    same_node = no_loop_result["src_nodes"] == no_loop_result["dst_nodes"]
    same_image = no_loop_result["images"].sum(dim=-1) == 0
    assert torch.sum(torch.logical_and(same_node, same_image)) == 0
    allow_loop_transform = PeriodicPropertiesTransform(
        cutoff_radius=6.0, backend="ase", allow_self_loops=True
    )
    loop_result = allow_loop_transform(packed_data)
    # there should be some self-loops in this graph
    same_node = loop_result["src_nodes"] == loop_result["dst_nodes"]
    same_image = loop_result["images"].sum(dim=-1) == 0
    assert torch.sum(torch.logical_and(same_node, same_image)) > 0
