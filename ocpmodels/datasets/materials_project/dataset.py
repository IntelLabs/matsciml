from typing import Iterable, Tuple, Any, Dict, Union

import torch
import numpy as np
from pymatgen.core import Structure

from ocpmodels.datasets.base import BaseOCPDataset


class MaterialsProjectDataset(BaseOCPDataset):
    def index_to_key(self, index: int) -> Tuple[int]:
        return (0, index)

    def data_from_key(
        self, lmdb_index: int, subindex: int
    ) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Extract data out of the PyMatGen data structure and format into PyTorch happy structures.

        In line with the rest of the framework, this method returns a nested
        dictionary. Specific to this format, however, we separate features and
        targets: at the top of level we expect what is effectively a point cloud
        with `coords` and `atomic_numbers`, while the `lattice_features` and
        `targets` keys nest additional values 

        Parameters
        ----------
        lmdb_index
            [TODO:description]
        subindex
            [TODO:description]

        Returns
        -------
        Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]
            [TODO:description]
        """
        data: Dict[str, Any] = super().data_from_key(lmdb_index, subindex)
        return_dict = {}
        structure: Structure = data.get("structure")
        # retrieve properties
        return_dict["coords"] = torch.from_numpy(structure.cart_coords).float()
        return_dict["atomic_numbers"] = torch.LongTensor(structure.atomic_numbers)
        # grab lattice properties
        space_group = structure.get_space_group_info()[-1]
        lattice_params = torch.FloatTensor(
            structure.lattice.abc + structure.lattice.angles
        )
        lattice_features = {
            "space_group": space_group,
            "lattice_params": lattice_params,
        }
        return_dict["lattice_features"] = lattice_features
        # assume every other key are targets
        not_targets = set(["structure", "fields_not_requested"] + data["fields_not_requested"])
        target_keys = set(data.keys()).difference(not_targets)
        targets = {key: self._standardize_values(data[key]) for key in target_keys}
        return_dict["targets"] = targets
        # compress all the targets into a single tensor for convenience
        target_tensor = []
        for key in target_keys:
            item = data.get(key)
            if isinstance(item, Iterable):
                target_tensor.extend(item)
            else:
                target_tensor.append(item)
        target_tensor = torch.FloatTensor(target_tensor)
        return_dict["target_tensor"] = target_tensor
        return return_dict

    @staticmethod
    def _standardize_values(value: Union[float, Iterable[float]]) -> Union[torch.Tensor, float]:
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
        else:
            # for scalars, just return the value
            return value

