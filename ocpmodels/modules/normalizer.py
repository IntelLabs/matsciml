"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
from typing import Union, Dict

import torch


class Normalizer(object):
    """Normalize a Tensor and restore it later."""

    def __init__(self, tensor=None, mean=None, std=None, device=None):
        """tensor is taken as a sample to calculate the mean and std"""
        if tensor is None and mean is None:
            return

        if device is None:
            device = "cpu"

        if tensor is not None:
            self.mean = torch.mean(tensor, dim=0).to(device)
            self.std = torch.std(tensor, dim=0).to(device)
            return

        if mean is not None and std is not None:
            self.mean = torch.tensor(mean).to(device)
            self.std = torch.tensor(std).to(device)

    def to(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {"mean": self.mean, "std": self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict["mean"].to(self.mean.device)
        self.std = state_dict["std"].to(self.mean.device)


class BatchScaler(Normalizer):
    def __init__(self, dim: int = 0) -> None:
        super().__init__(mean=1.)
        self.dim = dim
        self.storage = {}

    def to(self, device: str) -> None:
        assert len(self.storage) > 0, f"No keys in scaler storage to move to {device}"
        new_storage_dict = {}
        for key, tensor in self.storage.items():
            new_storage_dict[key] = tensor.to(device)

    def norm(self, tensor: torch.Tensor, name: str) -> torch.Tensor:
        mean = tensor.mean(self.dim)
        std = tensor.std(self.std)
        self.storage[f"{name}_mean"] = mean
        self.storage[f"{name}_std"] = std
        return (tensor - mean) / std

    def denorm(self, normed_tensor: torch.Tensor, name: str) -> torch.Tensor:
        # make sure we only denorm tensors with normalized values
        mean, std = self.storage.get(f"{name}_mean", None), self.storage.get(f"{name}_std", None)
        if mean is None or std is None:
            raise ValueError(f"No normalized value stored for {name}!")
        return normed_tensor * std + mean

    def state_dict(self) -> Dict[str, Union[torch.Tensor, float]]:
        return self.storage
