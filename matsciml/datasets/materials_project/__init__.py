from __future__ import annotations

from importlib.util import find_spec

from matsciml.datasets.materials_project.api import MaterialsProjectRequest
from matsciml.datasets.materials_project.dataset import (
    MaterialsProjectDataset,
    MaterialsTrajectoryDataset,
)

__all__ = [
    "MaterialsProjectDataset",
    "MaterialsTrajectoryDataset",
    "MaterialsProjectRequest",
]

if find_spec("torch_geometric") is not None:
    from matsciml.datasets.materials_project.dataset import (
        CdvaeLMDBDataset,
        PyGCdvaeDataset,
        PyGMaterialsProjectDataset,
    )

    __all__.extend(
        ["CdvaeLMDBDataset", "PyGCdvaeDataset", "PyGMaterialsProjectDataset"]
    )
