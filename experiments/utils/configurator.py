from __future__ import annotations

from typing import Any
from collections import defaultdict

from pathlib import Path
import yaml


class Configurator:
    def __init__(self):
        self.datasets = {}
        self.models = {}
        self.trainer = {}

    @staticmethod
    def get_yaml_files(path: Path) -> list[Path]:
        files = []
        if path.is_file():
            files.append(path)
        else:
            for filename in path.rglob("*.yaml"):
                files.append(filename)
        return files

    @staticmethod
    def get_yaml_file_content(files: list[Path]) -> dict[str, Any]:
        content = defaultdict(dict)
        for filename in files:
            with open(filename, "r") as file:
                data = yaml.safe_load(file)
                file_key = filename.stem
                content[file_key].update(data)
        return content

    def configure_models(self, path: Path) -> dict[str, Any]:
        available_models = {
            "generic": {
                "output_kwargs": {
                    "norm": {
                        "class_path": "torch.nn.LayerNorm",
                        "init_args": {"normalized_shape": 128},
                    },
                    "hidden_dim": 128,
                    "activation": "SiLU",
                    "lazy": False,
                    "input_dim": 128,
                },
                "lr": 0.0001,
            },
        }
        content = self.extract_content(path)
        available_models.update(content)
        self.models = available_models

    def configure_datasets(self, path: Path) -> dict[str, Any]:
        available_data = {
            "generic": {
                "experiment": {"batch_size": 32, "num_workers": 16},
                "debug": {"batch_size": 4, "num_workers": 0},
            },
        }
        content = self.extract_content(path)
        available_data.update(content)
        self.datasets = available_data

    def configure_trainer(self, path: Path) -> dict[str, Any]:
        trainer_args = {
            "generic": {"min_epochs": 15, "max_epochs": 100},
        }
        content = self.extract_content(path)
        for key in content.keys():
            trainer_args.update(content[key])

        self.trainer = trainer_args

    def extract_content(self, path):
        yaml_files = Configurator.get_yaml_files(path)
        content = Configurator.get_yaml_file_content(yaml_files)
        return content

    def add_model(self, model_dict: dict[str, Any]) -> None:
        self.models.update(model_dict)

    def add_dataset(self, dataset_dict: dict[str, Any]) -> None:
        self.models.update(dataset_dict)


configurator = Configurator()
