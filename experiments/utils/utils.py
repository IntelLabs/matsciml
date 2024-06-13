from typing import Any

import pytorch_lightning  # noqa: F401
import matsciml  # noqa: F401


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
