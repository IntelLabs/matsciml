from __future__ import annotations
from functools import partial
from typing import Callable, Literal

import torch
from torch import nn


__all__ = ["AtomWeightedL1", "AtomWeightedMSE"]


class AtomWeightedL1(nn.Module):
    """
    Calculates the L1 loss between predicted and targets,
    weighted by the number of atoms within each graph.
    """

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        atoms_per_graph: torch.Tensor,
    ) -> torch.Tensor:
        if atoms_per_graph.size(0) != target.size(0):
            raise RuntimeError(
                "Dimensions for atom-weighted loss do not match:"
                f" expected atoms_per_graph to have {target.size(0)} elements; got {atoms_per_graph.size(0)}."
                "This loss is intended to be applied to scalar targets only."
            )
        # check to make sure we are broad casting correctly
        if (input.ndim != target.ndim) and target.size(-1) == 1:
            input.unsqueeze_(-1)
        # for N-d targets, we might want to keep unsqueezing
        while atoms_per_graph.ndim < target.ndim:
            atoms_per_graph.unsqueeze_(-1)
        # ensures that atoms_per_graph is type cast correctly
        squared_error = ((input - target) / atoms_per_graph.to(input.dtype)).abs()
        return squared_error.mean()


class AtomWeightedMSE(nn.Module):
    """
    Calculates the mean-squared-error between predicted and targets,
    weighted by the number of atoms within each graph.
    """

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        atoms_per_graph: torch.Tensor,
    ) -> torch.Tensor:
        if atoms_per_graph.size(0) != target.size(0):
            raise RuntimeError(
                "Dimensions for atom-weighted loss do not match:"
                f" expected atoms_per_graph to have {target.size(0)} elements; got {atoms_per_graph.size(0)}."
                "This loss is intended to be applied to scalar targets only."
            )
        # check to make sure we are broad casting correctly
        if (input.ndim != target.ndim) and target.size(-1) == 1:
            input.unsqueeze_(-1)
        # for N-d targets, we might want to keep unsqueezing
        while atoms_per_graph.ndim < target.ndim:
            atoms_per_graph.unsqueeze_(-1)
        # ensures that atoms_per_graph is type cast correctly
        squared_error = ((input - target) / atoms_per_graph.to(input.dtype)) ** 2.0
        return squared_error.mean()


class BatchQuantileLoss(nn.Module):
    def __init__(
        self,
        quantile_weights: dict[float, float],
        loss_func: Callable | Literal["mse", "rmse", "huber"],
        use_norm: bool = True,
        huber_delta: float | None = None,
    ) -> None:
        super().__init__()
        for key, value in quantile_weights.items():
            assert isinstance(
                key, float
            ), "Expected quantile keys to be floats between [0,1]."
            assert isinstance(
                value, float
            ), "Expected quantile dict values to be floats."
            assert (
                0.0 <= key <= 1.0
            ), f"Quantile value {key} invalid; must be between [0,1]."
        quantiles = torch.Tensor(list(quantile_weights.keys()))
        self.register_buffer("quantiles", quantiles)
        weights = torch.Tensor(list(quantile_weights.values()))
        self.register_buffer("weights", weights)
        self.use_norm = use_norm
        # each loss is wrapped as a partial to provide static arguments, primarily
        # as we want to not apply the reduction immediately
        if isinstance(loss_func, str):
            if loss_func == "mse":
                loss_func = partial(torch.nn.functional.mse_loss, reduction="none")
            elif loss_func == "rmse":
                raise NotImplementedError("RMSE function has not yet been implemented.")
            elif loss_func == "huber":
                assert (
                    huber_delta
                ), "Huber loss specified but no margin provided to ``huber_delta``."
                loss_func = partial(
                    torch.nn.functional.huber_loss, delta=huber_delta, reduction="none"
                )
        self.loss_func = loss_func

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.use_norm:
            target_quantity = target.norm(dim=-1, keepdim=True)
        else:
            target_quantity = target
        target_quantiles = torch.quantile(target_quantity, q=self.quantiles)
        target_weights = torch.empty_like(target_quantity)
        # define the first quantile bracket
        target_weights[target_quantity < target_quantiles[0]] = self.weights[0]
        # now do quantiles in between
        for index in range(1, len(self.weights) - 1):
            curr_quantile = self.quantiles[index]
            next_quantile = self.quantiles[index + 1]
            curr_weight = self.weights[index]
            mask = (target_quantity >= curr_quantile) & (
                target_quantity < next_quantile
            )
            target_weights[mask] = curr_weight
        # the last bin
        target_weights[target_quantity > target_quantiles[-1]] = self.weights[-1]
        unweighted_loss = self.loss_func(input, target)
        weighted_loss = unweighted_loss * target_weights
        return weighted_loss.mean()
