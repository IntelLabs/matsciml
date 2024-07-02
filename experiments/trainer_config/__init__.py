from experiments.trainer_config.trainer_config import setup_trainer  # noqa: F401

import os
import yaml


yaml_dir = yaml_dir = os.path.dirname(os.path.abspath(__file__))
trainer_args = {
    "generic": {"min_epochs": 15, "max_epochs": 100},
}

for filename in os.listdir(yaml_dir):
    if filename.endswith(".yaml"):
        file_path = os.path.join(yaml_dir, filename)
        with open(file_path, "r") as file:
            content = yaml.safe_load(file)
            file_key = os.path.splitext(filename)[0]
            trainer_args.update(content)
