from typing import Any, Union
import os

import matsciml  # noqa: F401
from matsciml.models.common import get_class_from_name
from matsciml.common.inspection import get_model_all_args


def verify_class_args(input_class, input_args):
    all_args = get_model_all_args(input_class)

    for key in input_args:
        assert (
            key in all_args
        ), f"{key} was passed as a kwarg but does not match expected arguments."


def instantiate_arg_dict(input: Union[list, dict[str, Any]]) -> dict[str, Any]:
    """Used to traverse through an config file and spin up any arguments that specify
    a 'class_path' and optional 'init_args'. Replaces the string values with the
    instantiated class. If the tag is a 'class_instance' this is simple a class which
    has not been instantiated yet.

    Parameters
    ----------
    input : dict[str, Any]
        Input config dictionary.

    Returns
    -------
    dict[str, Any]
        Updated config dictionary with instantiated classes as necessary.
    """
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
                verify_class_args(class_path, transform_args)
                return class_path(**transform_args)
            if key == "encoder_class":
                input[key] = get_class_from_name(value["class_path"])
            elif isinstance(value, dict) and "class_path" in value:
                class_path = value["class_path"]
                class_path = get_class_from_name(class_path)
                input_args = value.get("init_args", {})
                verify_class_args(class_path, input_args)
                input[key] = class_path(**input_args)
            else:
                input[key] = instantiate_arg_dict(value)
    elif isinstance(input, list):
        for i, item in enumerate(input):
            input[i] = instantiate_arg_dict(item)
    return input


def setup_log_dir(config: dict[str, Any]) -> str:
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


def _get_next_version(root_dir: str) -> str:
    if not os.path.isdir(root_dir):
        os.makedirs(root_dir)

    existing_versions = []
    for d in os.listdir(root_dir):
        if os.path.isdir(os.path.join(root_dir, d)) and d.startswith("version_"):
            existing_versions.append(int(d.split("_")[1]))

    if len(existing_versions) == 0:
        return "version_0"

    return f"version_{max(existing_versions) + 1}"


def convert_string(input_str: str) -> Union[int, float, str]:
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
) -> dict[str, Any]:
    """Update a config with arguments supplied from the cli. Can only update
    to numeric or string values by dictionary keys. Lists such as callbacks, loggers,
    or transforms are not updatable.

    Example:

    dict_name = "dataset"
    arg_dict = {'debug': {'batch_size': 4, 'num_workers': 0}}
    new_args = [['dataset', 'debug', 'batch_size', '20']]

    The input specifies that we are updating the arg_dict with new_args affecting the
    'dataset' config.
    The dictionary keys to traverse through will be "debug" and "batch_size".
    The target value to update to is '20', which will be converted to an int.


    Parameters
    ----------
    dict_name : str
        Dictionary to be updated, (model, dataset, or trainer)
    arg_dict : dict[str, Any]
        Original dictionary
    new_args : list[list[str]]
        Lists of arguments used to specify dictionary to update, the arguments to
        traverse through to update, and the value to update to.

    Returns
    -------
    dict[str, Any]
        New arg_dict with updated parameters.
    """
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


def config_help() -> None:
    from experiments.datasets import available_data
    from experiments.models import available_models

    print("Models:")
    _ = [print("\t", m) for m in available_models.keys() if m != "generic"]
    print()
    print("Datasets and Target Keys:")
    for k, v in available_data.items():
        if k != "generic":
            print(f"\t{k}: {v['target_keys']}")
