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
            cls.mapping["tasks"][name] = func
            return func

        return wrap

    @classmethod
    def register_dataset(cls: Registry, name: str):
        def wrap(func):
            cls.mapping["datasets"][name] = func
            return func

        return wrap

    @classmethod
    def register_datamodule(cls: Registry, name: str):
        def wrap(func):
            cls.mapping["datamodules"][name] = func
            return func

        return wrap

    @classmethod
    def register_model(cls: Registry, name: str):
        def wrap(func):
            cls.mapping["models"][name] = func
            return func

        return wrap

    @classmethod
    def register_transform(cls: Registry, name: str):
        def wrap(func):
            cls.mapping["transforms"][name] = func
            return func

        return wrap

    @classmethod
    def get_task_class(cls: Registry, name: str):
        return cls.mapping["tasks"].get(name, None)

    @classmethod
    def get_dataset_class(cls: Registry, name: str):
        return cls.mapping["datasets"].get(name, None)

    @classmethod
    def get_datamodule_class(cls: Registry, name: str):
        return cls.mapping["datamodules"].get(name, None)

    @classmethod
    def get_model_class(cls: Registry, name: str):
        return cls.mapping["models"].get(name, None)

    @classmethod
    def get_transform_class(cls: Registry, name: str):
        return cls.mapping["models"].get(name, None)

    @classmethod
    def get(cls, name, default=None, no_warning=False):
        r"""Get an item from registry with key 'name'

        Args:
            name (string): Key whose value needs to be retreived.
            default: If passed and key is not in registry, default value will
                     be returned with a warning. Default: None
            no_warning (bool): If passed as True, warning when key doesn't exist
                               will not be generated. Useful for cgcnn's
                               internal operations. Default: False
        Usage::

            from ocpmodels.common.registry import registry

            config = registry.get("config")
        """
        original_name = name
        name = name.split(".")
        value = cls.mapping["state"]
        for subname in name:
            value = value.get(subname, default)
            if value is default:
                break

        if (
            "writer" in cls.mapping["state"]
            and value == default
            and no_warning is False
        ):
            cls.mapping["state"]["writer"].write(
                "Key {} is not present in registry, returning default value "
                "of {}".format(original_name, default)
            )
        return value


registry = Registry()
