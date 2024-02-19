from __future__ import annotations

from matsciml.datasets import IS2REDataset, NomadDataset
from matsciml.datasets.transforms import (
    PeriodicPropertiesTransform,
    PointCloudToGraphTransform,
)

"""
This example shows how periodic boundary conditions can be wired
into the graphs via the transform pipeline interface.

We chain the `PeriodicPropertiesTransform`, which calculates the
offsets and images using Pymatgen, which provides the edge definitions
that are used by `PointCloudToGraphTransform`.
"""

dset = IS2REDataset.from_devset(
    transforms=[
        PeriodicPropertiesTransform(cutoff_radius=6.5),
        PointCloudToGraphTransform(backend="dgl"),
    ],
)

dset.__getitem__(25)
