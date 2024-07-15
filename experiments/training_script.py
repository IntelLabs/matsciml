import os
import yaml
from typing import Any

from experiments.datasets.data_module_config import setup_datamodule
from experiments.task_config.task_config import setup_task
from experiments.trainer_config.trainer_config import setup_trainer
from experiments.trainer_config import trainer_args

from experiments.utils.utils import setup_log_dir, config_help

from argparse import ArgumentParser


def main(config: dict[str, Any]) -> None:
    os.makedirs(config["log_path"], exist_ok=True)

    dm = setup_datamodule(config)
    task = setup_task(config)
    trainer = setup_trainer(config, trainer_args=trainer_args)
    trainer.fit(task, datamodule=dm)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-o",
        "--options",
        help="Show options for models, datasets, and targets",
        action="store_true",
    )
    parser.add_argument(
        "-d",
        "--debug",
        help="Uses debug config with devsets and only a few batches per epoch.",
        action="store_true",
    )
    parser.add_argument(
        "-e",
        "--experiment_config",
        help="Experiment config yaml file to use.",
    )
    parser.add_argument(
        "-c",
        "--cli_args",
        nargs="+",
        help="Parameters to update via cli, such as: dataset.debug.batch_size.16",
        default=None,
    )
    args = parser.parse_args()
    if args.options:
        config_help()
        os._exit(0)
    config = yaml.safe_load(open(args.experiment_config))
    config["cli_args"] = (
        [arg.split(".") for arg in args.cli_args] if args.cli_args else None
    )
    log_path = setup_log_dir(config)
    config["log_path"] = log_path
    config["run_type"] = run_type = "debug" if args.debug else "experiment"
    main(config)
