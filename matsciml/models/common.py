from __future__ import annotations

from copy import deepcopy
from importlib import import_module
from inspect import getfullargspec
from typing import Any, Callable

import torch
from e3nn import nn as e3layers
from e3nn import o3
from torch import nn
from torch.nn.parameter import Parameter

from matsciml.common.registry import registry


def get_class_from_name(class_path: str) -> type[Any]:
    """
    Load in a specified module, and retrieve a class within
    that module.

    The main use case for this function is to convert a class
    path into the actual class itself, in a way that doesn't
    allow arbitrary code to be executed by the user.

    Parameters
    ----------
    class_path : str
        String representation of the class path, such as `torch.nn.SiLU`

    Returns
    -------
    Type[Any]
        Loaded class reference
    """
    split_str = class_path.split(".")
    module_str = ".".join(split_str[:-1])
    class_str = split_str[-1]
    module = import_module(module_str)
    return getattr(module, class_str)


@registry.register_model("OutputBlock")
class OutputBlock(nn.Module):
    """
    Building block for output heads. Simple MLP stack with
    option to include residual connections.
    """

    def __init__(
        self,
        output_dim: int,
        activation: nn.Module | type[nn.Module] | Callable | str | None = None,
        norm: nn.Module | type[nn.Module] | Callable | str | None = None,
        input_dim: int | None = None,
        lazy: bool = True,
        bias: bool = True,
        dropout: float = 0.0,
        residual: bool = True,
    ) -> None:
        """
        Initialize an `OutputBlock` MLP.

        This model uses `LazyLinear` layers to create uninitialized MLPs,
        which means no input dimensionality is needed to be specified.

        Parameters
        ----------
        output_dim : int
            Dimensionality of the output of this model.
        activation : Optional[Union[nn.Module, Type[nn.Module], Callable, str]], default None
            If None, uses `nn.Identity()` as a placeholder. This nonlinearity is applied
            before normalization.
        norm : Optional[Union[nn.Module, Type[nn.Module], Callable, str]], default None
            If None, uses `nn.Identity()` as a placeholder. This applies some normalization
            between hidden layers, after activation.
        dropout : float, default 0.
            Probability of dropout in hidden layers.
        residual : bool, default True
            Flag to specify whether residual connections are used between
            hidden layer.
        """
        super().__init__()
        if activation is None:
            activation = nn.Identity
        if isinstance(activation, str):
            activation = get_class_from_name(activation)
        if isinstance(activation, type):
            activation = activation()
        if norm is None:
            norm = nn.Identity
        if isinstance(norm, str):
            norm = get_class_from_name(norm)
        if isinstance(norm, type):
            norm = norm()
        self.residual = residual
        if lazy:
            linear = nn.LazyLinear(output_dim, bias=bias)
        else:
            if not lazy and not input_dim:
                raise ValueError(
                    "Non-lazy model specified for 'OutputBlock', but no 'input_dim' was passed.",
                )
            linear = nn.Linear(input_dim, output_dim, bias=bias)
        dropout = nn.Dropout(dropout)
        # be liberal about deepcopy, to make sure we don't duplicate weights
        # when we don't intend to
        self.layers = nn.Sequential(
            linear,
            deepcopy(activation),
            deepcopy(norm),
            deepcopy(dropout),
        )

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        output = self.layers(data)
        if self.residual:
            assert output.shape == data.shape
            output = output + data
        return output

    @property
    def input_dim(self) -> int:
        """
        Return the expected input size of this ``OutputBlock``.

        Returns
        -------
        int
            ``nn.Linear`` weight matrix size
        """
        return self.layers[0].weight.size(-1)


@registry.register_model("IrrepOutputBlock")
class IrrepOutputBlock(nn.Module):
    def __init__(
        self,
        output_dim: str | o3.Irreps,
        input_dim: str | o3.Irreps,
        activation: list[str | e3layers.Activation | None] | None = None,
        norm: e3layers.BatchNorm | nn.Module | bool = True,
        residual: bool = True,
        **kwargs,
    ) -> None:
        """
        Initialize an `IrrepOutputBlock` MLP.

        This output projection block is intended to preserve irreducible representations
        that are created through equivariant neural networks.

        Parameters
        ----------
        output_dim : str | o3.Irreps
            Irreducible representations of the output projection
        input_dim : str | o3.Irreps
            Irreducible representations of the input representation
        activation : Optional[list[str | e3.nn.Activation | None]], default None
            A sequence of activation functions to apply to the output projection.
        norm : Optional[e3.nn.BatchNorm | nn.Module | bool], default True
            Applies normalization after non-linearity. Default value of ``True``
            will automatically use `e3.nn.BatchNorm` with the correct representations.
        residual : bool, default True
            Flag to specify whether residual connections are used between
            hidden layer.
        """
        kwargs.setdefault("biases", True)
        super().__init__()
        self.residual = residual
        if not isinstance(output_dim, o3.Irreps):
            output_dim = o3.Irreps(output_dim)
        if not isinstance(input_dim, o3.Irreps):
            input_dim = o3.Irreps(input_dim)
        # before mapping kwargs, filter out bad incorrect ones
        linear_sig = getfullargspec(o3.Linear)
        linear_args = set(linear_sig.args) | set(linear_sig.kwonlyargs)
        kwargs = {key: value for key, value in kwargs.items() if key in linear_args}
        linear = o3.Linear(input_dim, output_dim, **kwargs)
        # only go through the process of making activation if it's specified
        if activation is not None:
            if not isinstance(activation, list):
                activation = [activation]
            for index, act in enumerate(activation):
                if isinstance(act, str):
                    act = get_class_from_name(act)
                # if we haven't instantiated the activation, do it now
                if isinstance(act, type) and act is not None:
                    act = act()
                activation[index] = act
            # make sure we have enough activation functions
            if len(activation) != len(output_dim):
                raise ValueError(
                    "Number of activations passed not equal to number of representations; "
                    f"got {len(activation)}, expected {len(output_dim)}",
                )
            # if we haven't converted the activation functions into the e3.nn wrapper,
            # do so now
            if not isinstance(activation, e3layers.Activation):
                activation = e3layers.Activation(irreps_in=output_dim, acts=activation)
        else:
            activation = nn.Identity()
        if isinstance(norm, bool):
            if norm:
                norm = e3layers.BatchNorm(output_dim)
        else:
            norm = nn.Identity()
        self.layers = nn.Sequential(linear, deepcopy(activation), deepcopy(norm))

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        output = self.layers(data)
        if self.residual:
            assert output.shape == data.shape
            output = output + data
        return output

    @property
    def input_dim(self) -> int:
        """
        Return the expected input size of this ``OutputBlock``.

        Returns
        -------
        int
            ``nn.Linear`` weight matrix size
        """
        return self.layers[0].weight.size(-1)


class OutputHead(nn.Module):
    """
    A stack of output blocks, constituting an output head.

    Action of this stack is to transform a common embedding into
    actual outputs. The default settings will correspond to an
    MLP without any nonlinearities or normalizations.
    """

    def __init__(
        self,
        output_dim: int | str,
        hidden_dim: int | str,
        num_hidden: int = 1,
        activation: nn.Module | type[nn.Module] | Callable | str | None = None,
        norm: nn.Module | type[nn.Module] | Callable | str | None = None,
        act_last: nn.Module | type[nn.Module] | Callable | str | None = None,
        input_dim: int | str | None = None,
        block_type: type[nn.Module] | str = OutputBlock,
        **kwargs,
    ) -> None:
        """
        Initialize an `OutputHead` architecture.

        This model uses `LazyLinear` layers to create uninitialized MLPs,
        which means no input dimensionality is needed to be specified.
        Kwargs are passed into the instantiation of ``block_type``.

        Parameters
        ----------
        output_dim : int | str
            Dimensionality of the output of this model. String inputs are
            specifically for ``IrrepOutputBlock``.
        hidden_dim : int | str
            Dimensionality of the hidden layers within this stack. String
            inputs are specifically for ``IrrepOutputBlock``.
        num_hidden : int
            Number of hidden `OutputBlock`s to use.
        input_dim : int | str | None, default None
            Dimensionality of input data into the output head; typically would be the
            embedding dimensionality. String inputs are specifically for ``IrrepOutputBlock``.
        activation : Optional[Union[nn.Module, Type[nn.Module], Callable, str]], default None
            If None, uses `nn.Identity()` as a placeholder. This nonlinearity is applied
            before normalization for every hidden layer within the stack.
        norm : Optional[Union[nn.Module, Type[nn.Module], Callable, str]], default None
            If None, uses `nn.Identity()` as a placeholder. This applies some normalization
            between hidden layers, after activation.
        act_last : Optional[Union[nn.Module, Type[nn.Module], Callable, str]], default None
            If None, uses `nn.Identity()` as a a placeholder. This is an optional output
            layer activation function.
        dropout : float, default 0.
            Probability of dropout in hidden layers.
        residual : bool, default True
            Flag to specify whether residual connections are used between
            hidden layer.
        block_type : type[nn.Module] | str, default ``OutputBlock``
            The type of block to constitute this output head. By default, the
            naive ``OutputBlock`` just performs MLP projections, whereas ``IrrepOutputBlock``
            will preserve irreducible representations in the output. If a String
            is passed, the class will be retrieved from the registry.
        """
        kwargs.setdefault("lazy", True)
        kwargs.setdefault("dropout", 0.0)
        kwargs.setdefault("residual", True)
        kwargs.setdefault("bias", True)
        super().__init__()
        if isinstance(block_type, str):
            type_name = block_type
            block_type = registry.get_model_class(block_type)
            if not block_type:
                raise NameError(
                    f"Specified block type {type_name} does not exist in matsciml.models.common.",
                )
        blocks = [
            block_type(
                output_dim=hidden_dim,
                activation=activation,
                norm=norm,
                input_dim=input_dim,
                **kwargs,
            ),
        ]
        # for everything in between
        blocks.extend(
            [
                block_type(
                    output_dim=hidden_dim,
                    activation=activation,
                    norm=norm,
                    input_dim=hidden_dim,
                    **kwargs,
                )
                for _ in range(num_hidden)
            ],
        )
        # last layer does not use residual or normalization
        kwargs["residual"] = False
        blocks.append(
            block_type(
                output_dim=output_dim,
                activation=act_last,
                norm=None,
                input_dim=hidden_dim,
                **kwargs,
            ),
        )
        self.blocks = nn.Sequential(*blocks)
        self.lazy = kwargs.get("lazy")

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        if not self.lazy:
            expected_shape = self.blocks[0].input_dim
            assert (
                embedding.size(-1) == expected_shape
            ), f"Incoming encoder output dim ({embedding.size(-1)}) does not match the expected 'OutputBlock' dim ({expected_shape})"
        return self.blocks(embedding)


class RMSNorm(nn.Module):
    """
    Original code by https://github.com/bzhangGo/rmsnorm/blob/master/rmsnorm_torch.py
    """

    def __init__(self, input_dim: int, eps: float = 1e-8, bias: bool = False) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.eps = eps

        self.scale = Parameter(torch.ones(input_dim))
        self.bias = Parameter(torch.zeros(input_dim)) if bias else nn.Identity()
        self.has_bias = bias
        self.register_parameter("scale", self.scale)
        if bias:
            self.register_parameter("bias", self.bias)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        tensor_norm: torch.Tensor = torch.norm(data, p=2, dim=-1, keepdim=True)
        rms_values = torch.sqrt(tensor_norm * self.input_dim)
        # apply the RMSNorm to inputs
        norm_output = (data / (rms_values + self.eps)) * self.scale
        if self.has_bias:
            norm_output = norm_output + self.bias
        return norm_output


class PartialRMSNorm(RMSNorm):
    """
    Original code by https://github.com/bzhangGo/rmsnorm/blob/master/rmsnorm_torch.py

    This implements the partial RMS norm as a separate class to improve readibility
    and maintainability.
    """

    def __init__(
        self,
        input_dim: int,
        eps: float = 1e-8,
        partial: float = 0.5,
        bias: bool = False,
    ) -> None:
        super().__init__(input_dim, eps, bias)
        self.partial = partial

    @property
    def partial(self) -> float:
        return self._partial

    @partial.setter
    def partial(self, value: float) -> None:
        assert (
            0.0 < value < 1.0
        ), f"Partial value must be in the range [0,1]; value: {value}"
        self._partial = value

    @property
    def partial_length(self) -> int:
        return int(self.partial * self.input_dim)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        # split the input data along partial
        split_tensor, _ = torch.split(
            data,
            [self.partial_length, self.input_dim - self.partial_length],
            dim=-1,
        )
        # compute norm based on the split portion
        tensor_norm: torch.Tensor = torch.norm(split_tensor, p=2, dim=1)
        rms_values = torch.sqrt(tensor_norm * self.partial_length)
        norm_output = (data / (rms_values + self.eps)) * self.scale
        if self.has_bias:
            norm_output = norm_output + self.bias
        return norm_output


class SymmetricLog(nn.Module):
    """
    Implements the ``SymmetricLog`` activation as described in
    Cai et al., https://arxiv.org/abs/2111.15631

    The activation is asymptotic and provides gradients over
    a large range of possible values.
    """

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        tanx = data.tanh()
        return tanx * torch.log(data * tanx + 1)
