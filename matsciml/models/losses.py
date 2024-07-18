from __future__ import annotations

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
