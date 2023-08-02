from copy import deepcopy
from functools import cached_property
from math import pi
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch

from ocpmodels.common.registry import registry
from ocpmodels.common.types import BatchDict, DataDict
from ocpmodels.datasets.base import PointCloudDataset
from ocpmodels.datasets.utils import (
    concatenate_keys,
    pad_point_cloud,
    point_cloud_featurization,
)


@registry.register_dataset("NomadDataset")
class NomadDataset(PointCloudDataset):
    __devset__ = Path(__file__).parents[0].joinpath("devset")

    def index_to_key(self, index: int) -> Tuple[int]:
        return (0, index)

    @staticmethod
    def collate_fn(batch: List[DataDict]) -> BatchDict:
        return concatenate_keys(
            batch,
            pad_keys=["pc_features"],
            unpacked_keys=["sizes", "src_nodes", "dst_nodes"],
        )

    def raw_sample(self, idx):
        return super().data_from_key(0, idx)
    

    @property
    def target_keys(self) -> Dict[str, List[str]]:
        return {"regression": ["energy"], "classification":["is_metal"]}

    @staticmethod
    def _standardize_values(
        value: Union[float, Iterable[float]]
    ) -> Union[torch.Tensor, float]:
        """
        Standardizes targets to be ingested by a model.

        For scalar values, we simply return it unmodified, because they can be easily collated.
        For iterables such as tuples and NumPy arrays, we use the appropriate tensor creation
        method, and typecasted into FP32 or Long tensors.

        The type hint `float` is used to denote numeric types more broadly.

        Parameters
        ----------
        value : Union[float, Iterable[float]]
            A target value, which can be a scalar or array of values

        Returns
        -------
        Union[torch.Tensor, float]
            Mapped torch.Tensor format, or a scalar numeric value
        """
        if isinstance(value, Iterable) and not isinstance(value, str):
            # get type from first entry
            dtype = torch.long if isinstance(value[0], int) else torch.float
            if isinstance(value, np.ndarray):
                return torch.from_numpy(value).type(dtype)
            else:
                return torch.Tensor(value).type(dtype)
        # for missing data, set to zero
        elif value is None:
            return 0.0
        else:
            # for scalars, just return the value
            return value


    def data_from_key(self, lmdb_index: int, subindex: int) -> Any:
        # for a full list of properties avaialbe: data['properties']['available_properties'
        data = super().data_from_key(lmdb_index, subindex)
        
        import pdb; pdb.set_trace()
        return_dict = {}
