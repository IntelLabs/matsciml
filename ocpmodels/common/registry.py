from __future__ import annotations

"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

# Copyright (c) Facebook, Inc. and its affiliates.
# Borrowed from https://github.com/facebookresearch/pythia/blob/master/pythia/common/registry.py.
"""
Registry is central source of truth. Inspired from Redux's concept of
global store, Registry maintains mappings of various information to unique
keys. Special functions in registry can be used as decorators to register
different kind of classes.

Import the global registry object using

``from ocpmodels.common.registry import registry``

Various decorators for registry different kind of classes with unique keys

- Register a model: ``@registry.register_model``
"""


class Registry:
    r"""Class for registry object which acts as central source of truth."""
    __entries__ = {
        "datasets": {},
        "datamodules": {},
        "models": {},
        "tasks": {},
        "transforms": {},
    }

    @classmethod
    def register_task(cls: Registry, name: str):
        def wrap(func):
            cls.__entries__["tasks"][name] = func
            return func

        return wrap

    @classmethod
    def register_dataset(cls: Registry, name: str):
        def wrap(func):
            cls.__entries__["datasets"][name] = func
            return func

        return wrap

    @classmethod
    def register_datamodule(cls: Registry, name: str):
        def wrap(func):
            cls.__entries__["datamodules"][name] = func
            return func

        return wrap

    @classmethod
    def register_model(cls: Registry, name: str):
        def wrap(func):
            cls.__entries__["models"][name] = func
            return func

        return wrap

    @classmethod
    def register_transform(cls: Registry, name: str):
        def wrap(func):
            cls.__entries__["transforms"][name] = func
            return func

        return wrap

    @classmethod
    def get_task_class(cls: Registry, name: str):
        return cls.__entries__["tasks"].get(name, None)

    @classmethod
    def get_dataset_class(cls: Registry, name: str):
        return cls.__entries__["datasets"].get(name, None)

    @classmethod
    def get_datamodule_class(cls: Registry, name: str):
        return cls.__entries__["datamodules"].get(name, None)

    @classmethod
    def get_model_class(cls: Registry, name: str):
        return cls.__entries__["models"].get(name, None)

    @classmethod
    def get_transform_class(cls: Registry, name: str):
        return cls.__entries__["transforms"].get(name, None)


registry = Registry()
