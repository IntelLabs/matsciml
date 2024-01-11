from __future__ import annotations

from importlib.util import find_spec
from pathlib import Path

from matsciml.datasets.materials_project.api import MaterialsProjectRequest
from matsciml.datasets.materials_project.dataset import MaterialsProjectDataset

if find_spec("torch_geometric") is not None:
    from matsciml.datasets.materials_project.dataset import (
        CdvaeLMDBDataset,
        PyGCdvaeDataset,
        PyGMaterialsProjectDataset,
    )

materialsproject_devset = Path(__file__).parents[0].joinpath("devset")
