import yaml

from torch.nn import LayerNorm


from pathlib import Path

yaml_dir = Path(__file__).parent
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

for filename in yaml_dir.rglob("*.yaml"):
    file_path = yaml_dir.joinpath(filename)
    with open(file_path, "r") as file:
        content = yaml.safe_load(file)
        file_key = file_path.stem
        available_models[file_key] = content
