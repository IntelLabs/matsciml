import os
import yaml


yaml_dir = yaml_dir = os.path.dirname(os.path.abspath(__file__))
available_data = {
    "generic": {
        "experiment": {"batch_size": 32, "num_workers": 16},
        "debug": {"batch_size": 2, "num_workers": 0},
    },
}


for filename in os.listdir(yaml_dir):
    if filename.endswith(".yaml"):
        file_path = os.path.join(yaml_dir, filename)
        with open(file_path, "r") as file:
            content = yaml.safe_load(file)
            file_key = os.path.splitext(filename)[0]
            available_data[file_key] = content
