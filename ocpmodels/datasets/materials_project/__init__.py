from pathlib import Path

from ocpmodels.datasets.materials_project.api import MaterialsProjectRequest
from ocpmodels.datasets.materials_project.dataset import MaterialsProjectDataset

materialsproject_devset = Path(__file__).parents[0].joinpath("devset")
