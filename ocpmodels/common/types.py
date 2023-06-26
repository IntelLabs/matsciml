from typing import Dict, Union

import torch

from ocpmodels.common import package_registry


# for point clouds
representations = [torch.Tensor]
if package_registry["pyg"]:
    from torch_geometric.data import Data as PyGGraph

    representations.append(PyGGraph)
if package_registry["dgl"]:
    from dgl import DGLGraph

    representations.append(DGLGraph)

representations = tuple(representations)

DataType = Union[representations]

# for a dictionary look up of data
DataDict = Dict[str, Union[float, DataType]]

# for a dictionary of batched data
BatchDict = Dict[str, Union[float, DataType, DataDict]]
