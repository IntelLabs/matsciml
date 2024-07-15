import os
import yaml

from pathlib import Path

yaml_dir = Path(__file__).parent


available_data = {
    "generic": {
        "experiment": {"batch_size": 32, "num_workers": 16},
        "debug": {"batch_size": 4, "num_workers": 0},
    },
}


for filename in yaml_dir.rglob("*.yaml"):
    file_path = Path(os.path.join(yaml_dir, filename))
    with open(file_path, "r") as file:
        content = yaml.safe_load(file)
        file_key = file_path.stem
        available_data[file_key] = content
