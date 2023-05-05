from schema import Schema
from copy import deepcopy
from typing import Union, Dict, Any, Type
from importlib import import_module
from pathlib import Path
from ruamel.yaml import YAML

from torch import nn
from pytorch_lightning.core.saving import save_hparams_to_yaml
from pytorch_lightning.utilities.parsing import AttributeDict


# for the most part, this schema just ensures there's the minimum
# number and types of keys in the config
task_schema = Schema(
    {
        "encoder": dict,
        "encoder_type": str,
        "task_hparams": dict,
        "task_type": str,
        "is_multitask": bool,
    },
    ignore_extra_keys=True,
)
# schema for config entries that need to be converted into types
type_schema = Schema({"module": str, "type": str}, ignore_extra_keys=False)


def get_hyperparameters(
    task: Union["BaseTaskModule", "MultiTaskLitModule"]
) -> Dict[str, Any]:
    params = {}
    encoder = task.encoder
    encoder_hparams = encoder.hparams
    # pack dictionary with all the goodies
    params["encoder"] = convert_hparams(encoder_hparams)
    params["encoder_type"] = get_object_path(encoder)
    params["task_hparams"] = convert_hparams(task.hparams)
    params["task_type"] = get_object_path(task)
    params["is_multitask"] = "MultiTask" in params["task_type"]
    # TODO add task output head weight initialization
    return params


def get_object_path(obj: object) -> str:
    obj_type = obj.__class__
    return f"{obj_type.__module__}.{obj_type.__name__}"


def get_class(path: str) -> Type:
    try:
        split_path = path.split(".")
        mod_path = ".".join(split_path[:-1])
        classname = split_path[-1]
        module = import_module(mod_path)
        return getattr(module, classname)
    except ImportError:
        raise ImportError(f"{path} is not a valid class or module is not installed.")


def convert_hparams(hparams: Union[dict, AttributeDict]):
    results = {}
    for key, value in hparams.items():
        if isinstance(value, (int, float)):
            results[key] = value
        elif isinstance(value, Type):
            results[key] = {"type": value.__name__, "module": value.__module__}
        elif isinstance(value, nn.Module):
            module_class = value.__class__
            module_origin = module_class.__module__
            results[key] = {"type": module_class.__name__, "module": module_origin}
        elif isinstance(value, dict):
            # recursively call function on dictionaries
            results[key] = convert_hparams(value)
    return results


def class_from_entry(dict_entry: Dict[str, str]) -> Type:
    """
    Map a dictionary entry into a Type.

    Parameters
    ----------
    dict_entry : Dict[str, str]
        Dictionary entry, expecting "module"

    Returns
    -------
    Type
        Class contained in `dict_entry`
    """
    type_schema.validate(dict_entry)
    ref = f"{dict_entry['module']}.{dict_entry['type']}"
    return get_class(ref)


def ingest_config(_dict: Dict[str, Any]) -> Dict[str, Any]:
    result = deepcopy(_dict)
    for key, value in _dict.items():
        if isinstance(value, dict):
            if "module" in value.keys():
                result[key] = class_from_entry(value)
            else:
                result[key] = ingest_config(value)
    return result


def reconstruct_from_yaml(
    yml_path: Union[Path, str]
) -> Union["BaseTaskModule", "MultiTaskLitModule"]:
    if isinstance(yml_path, str):
        yml_path = Path(yml_path)
    assert yml_path.exists(), f"Target YAML file is missing; passed {yml_path}"
    yaml = YAML()
    with open(yml_path, "r") as read_file:
        config = yaml.load(read_file)
    task_schema.validate(config)
    encoder_type = get_class(config["encoder_type"])
    # call function to recursively convert entries into types, and other
    # processing as needed
    encoder_config = ingest_config(config["encoder"])
    encoder = encoder_type(**encoder_config)
    task_type = get_class(config["task_type"])
    task_hparams = ingest_config(config["task_hparams"])
    return task_type(encoder, **task_hparams)
