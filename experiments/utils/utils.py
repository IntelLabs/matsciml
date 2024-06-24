from typing import Any
import os

import pytorch_lightning  # noqa: F401
import matsciml  # noqa: F401
from matsciml.models import *  # noqa: F401
from matsciml.datasets.transforms import *  # noqa: F401
from matsciml.lightning.callbacks import *  # noqa: F401


def instantiate_arg_dict(input: dict[str, Any]) -> dict[str, Any]:
    if isinstance(input, dict):
        for key, value in list(input.items()):
            if key == "class_path":
                class_path = value
                transform_args = {}
                input_args = input.get("init_args", {})
                if isinstance(input_args, list):
                    for input in input_args:
                        transform_args.update(input)
                else:
                    transform_args = input_args
                return eval(f"{class_path}(**{transform_args})")
            if key == "encoder_class":
                input[key] = eval(f"{value['class_path']}")
            elif isinstance(value, dict) and "class_path" in value:
                class_path = value["class_path"]
                input_args = value.get("init_args", {})
                input[key] = eval(f"{class_path}(**{input_args})")
            else:
                input[key] = instantiate_arg_dict(value)
    elif isinstance(input, list):
        for i, item in enumerate(input):
            input[i] = instantiate_arg_dict(item)
    return input


def setup_log_dir(config):
    model = config["model"]
    datasets = "_".join(list(config["dataset"].keys()))
    experiment_name = "_".join([model, datasets])
    if "log_dir" in config:
        log_dir = os.path.join(config["log_dir"], experiment_name)
    else:
        log_dir = os.path.join("experiment_logs", experiment_name)
    next_version = _get_next_version(log_dir)
    log_dir = os.path.join(log_dir, next_version)
    return log_dir


def _get_next_version(root_dir):
    if not os.path.isdir(root_dir):
        os.makedirs(root_dir)

    existing_versions = []
    for d in os.listdir(root_dir):
        if os.path.isdir(os.path.join(root_dir, d)) and d.startswith("version_"):
            existing_versions.append(int(d.split("_")[1]))

    if len(existing_versions) == 0:
        return "version_0"

    return f"version_{max(existing_versions) + 1}"
