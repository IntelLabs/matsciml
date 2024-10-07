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
        loss_func: Callable | Literal["mse", "mae", "rmse", "huber"],
        use_norm: bool = True,
        huber_delta: float | None = None,
    ) -> None:
        """
        Implements a batch-based or dynamic quantile loss function.

        This loss function uses user-defined quantiles and associated
        weights for training: the high-level idea is to allow flexibility
        in optimizing model performance against certain outliers, and
        ensuring that the model generalizes well.

        The function will either use the target values, or the norm of
        the target values (more meaningful for vector quantities like
        forces) to compute quantile values based on bins requested. A weight
        tensor is then generated (with the same shape as the targets) to
        weight predicted vs. actual margins, as computed with ``loss_func``.
        The mean of the weighted loss is then returned.

        Parameters
        ----------
        quantile_weights : dict[float, float]
            Dictionary mapping of quantile and the weighting to ascribe
            to that bin. Values smaller than the first bin, and larger
            than the last bin take on these respective values, while
            quantile in between bin ranges include the lower quantile
            and up to (not including) the next bin.
        loss_func : Callable | Literal['mse', 'mae', 'rmse', 'huber']
            Actual metric function. If a string literal is given, then
            one of the built-in PyTorch functional losses are used
            based on either MSE, RMSE, or Huber loss. If a ``Callable``
            is passed, the output **must** be of the same dimension
            as the targets, i.e. the behavior of ``keepdim`` or no
            reduction, as the weights are applied afterwards.
        use_norm : bool, default True
            Whether to use the norm of targets, instead of an elementwise
            wise application. This makes sense for vector quantities that
            are coupled, e.g. force vectors. If ``False``, this will still
            work with scalar and vector quantities-alike, but requires an
            intuition for one over the other.
        huber_delta : float, optional
            If ``loss_func`` is set to 'huber', this value is used as the
            ``delta`` argument in ``torch.nn.functional.huber_loss``, which
            corresponds to the margin between MAE/L1 and MSE functions.

        Raises
        ------
        NotImplementedError:
            Currently RMSE is not implemented, and will trigger this
            exception.
        """
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
            elif loss_func == "mae":
                loss_func = partial(torch.nn.functional.l1_loss, reduction="none")
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
            if target.ndim == 1:
                temp_target = target.unsqueeze(-1)
            else:
                temp_target = target
            target_quantity = temp_target.norm(dim=-1, keepdim=True)
        else:
            target_quantity = target
        target_quantiles = torch.quantile(target_quantity, q=self.quantiles)
        target_weights = torch.empty_like(target_quantity)
        # define the first quantile bracket
        target_weights[target_quantity < target_quantiles[0]] = self.weights[0]
        # now do quantiles in between
        for index in range(len(self.weights) - 1):
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
