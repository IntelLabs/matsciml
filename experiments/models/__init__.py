import os
import yaml

from torch.nn import LayerNorm


yaml_dir = yaml_dir = os.path.dirname(os.path.abspath(__file__))
available_models = {
    "generic": {
        "output_kwargs": {
            "norm": LayerNorm(128),
            "hidden_dim": 128,
            "activation": "SiLU",
            "lazy": False,
            "input_dim": 128,
        },
        "lr": 0.0001,
    },
}


for filename in os.listdir(yaml_dir):
    if filename.endswith(".yaml"):
        file_path = os.path.join(yaml_dir, filename)
        with open(file_path, "r") as file:
            content = yaml.safe_load(file)
            file_key = os.path.splitext(filename)[0]
            available_models[file_key] = content
