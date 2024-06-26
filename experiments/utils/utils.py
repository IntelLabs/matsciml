from typing import Any
import os

import matsciml  # noqa: F401
from matsciml.models.common import get_class_from_name


def instantiate_arg_dict(input: dict[str, Any]) -> dict[str, Any]:
    if isinstance(input, dict):
        for key, value in list(input.items()):
            if key == "class_instance":
                return get_class_from_name(value)
            if key == "class_path":
                class_path = value
                transform_args = {}
                input_args = input.get("init_args", {})
                if isinstance(input_args, list):
                    for input in input_args:
                        transform_args.update(input)
                else:
                    transform_args = input_args
                class_path = get_class_from_name(class_path)
                return class_path(**transform_args)
            if key == "encoder_class":
                input[key] = eval(f"{value['class_path']}")
            elif isinstance(value, dict) and "class_path" in value:
                class_path = value["class_path"]
                class_path = get_class_from_name(class_path)
                input_args = value.get("init_args", {})
                input[key] = class_path(**input_args)
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


def convert_string(input_str):
    # not sure if there is a better way to do this
    try:
        return int(input_str)
    except ValueError:
        pass
    try:
        return float(input_str)
    except ValueError:
        pass
    return input_str


def update_arg_dict(
    dict_name: str, arg_dict: dict[str, Any], new_args: list[list[str]]
):
    if new_args is None:
        return arg_dict
    updated_arg_dict = arg_dict
    new_args = [arg_list for arg_list in new_args if dict_name in arg_list]
    for new_arg in new_args:
        value = new_arg[-1]
        for key in new_arg[1:-1]:
            if key not in updated_arg_dict:
                updated_arg_dict[key] = {}
            if key != new_arg[-2]:
                updated_arg_dict = updated_arg_dict[key]
        updated_arg_dict[key] = convert_string(value)
    return arg_dict


def config_help():
    from experiments.datasets import available_data
    from experiments.models import available_models

    print("Models:")
    _ = [print("\t", m) for m in available_models.keys() if m != "generic"]
    print()
    print("Datasets and Target Keys:")
    for k, v in available_data.items():
        if k != "generic":
            print(f"\t{k}: {v['target_keys']}")
