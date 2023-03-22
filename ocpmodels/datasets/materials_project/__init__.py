from importlib.util import find_spec

from ocpmodels.datasets.materials_project.api import MaterialsProjectRequest
from ocpmodels.datasets.materials_project.dataset import MaterialsProjectDataset

if find_spec("dgl") is not None:
    from ocpmodels.datasets.materials_project.dataset import DGLMaterialsProjectDataset
