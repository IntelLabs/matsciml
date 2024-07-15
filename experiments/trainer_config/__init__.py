from experiments.trainer_config.trainer_config import setup_trainer  # noqa: F401

import os
import yaml
from pathlib import Path


yaml_dir = Path(__file__).parent
trainer_args = {
    "generic": {"min_epochs": 15, "max_epochs": 100},
}

for filename in yaml_dir.rglob("*.yaml"):
    file_path = Path(os.path.join(yaml_dir, filename))
    with open(file_path, "r") as file:
        content = yaml.safe_load(file)
        file_key = file_path.stem
        trainer_args.update(content)
