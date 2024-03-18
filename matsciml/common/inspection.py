# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: MIT License
from __future__ import annotations

from inspect import signature
from typing import Callable, Type

from torch import nn

"""
Simple utility functions for inspecting functions and objects.

The idea is that this should provide reusable functions that
are useful for mapping kwargs onto classes that belong outside
of this library, where we might not know what is required or not.
"""


def get_args_without_defaults(func: Callable, exclude_self: bool = True) -> list[str]:
    """
    Inspect a function for required positional input arguments.

    The function works by looping through arguments and checking
    if defaults are available. The option ``exclude_self`` is also
    available to specify whether or not to remove ``self`` entries
    from the list, since this may correspond to class methods.

    Parameters
    ----------
    func : Callable
        A callable function with some input arguments.
    exclude_self
        If True, ignores ``self`` and ``cls`` from the returned
        list.

    Returns
    -------
    list[str]
        List of argument names that are required by ``func``.
    """
    parameters = signature(func).parameters
    matches = []
    for arg_name, parameter in parameters.items():
        if getattr(parameter, "default", None):
            matches.append(arg_name)
    # remove self from list if requested
    if exclude_self:
        matches = list(filter(lambda x: x not in ["self", "cls"], matches))
    return matches


def get_all_args(func: Callable) -> list[str]:
    """
    Get all the arguments of a function, include with defaults.

    Parameters
    ----------
    func : Callable
        Function to inspect arguments for.

    Returns
    -------
    list[str]
        List of argument names, positional and keyword.
    """
    parameters = signature(func).parameters
    return list(parameters.keys())


def get_model_required_args(model: Type[nn.Module]) -> list[str]:
    """
    Inspect a child of PyTorch ``nn.Module`` for required arguments.

    The idea behind this is to identify which parameters are needed to
    instantiate a model, which is useful for determining how args/kwargs
    should be unpacked into a model from an abstract interface.

    Parameters
    ----------
    model : Type[nn.Module]
        A model class to inspect

    Returns
    -------
    list[str]
        List of argument names that are required to instantiate ``model``
    """
    return get_args_without_defaults(model.__init__, exclude_self=True)


def get_model_all_args(model: Type[nn.Module]) -> list[str]:
    """
    Inspect a model for all of its initialization arguments, including
    optional ones.

    Parameters
    ----------
    model : Type[nn.Module]
        Model class to inspect for arguments.

    Returns
    -------
    list[str]
        List of arguments used by the model instantiation.
    """
    return get_all_args(model.__init__)


def get_model_forward_args(model: Type[nn.Module]) -> list[str]:
    """
    Inspect a model's ``forward`` method for argument names.

    Parameters
    ----------
    model : Type[nn.Module]
        Model to inspect.

    Returns
    -------
    list[str]
        List of argument names in a model's forward method.
    """
    return get_all_args(model.forward)
