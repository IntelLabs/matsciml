import os
import yaml

from experiments.datasets.data_module_config import setup_datamodule
from experiments.task_config.task_config import setup_task
from experiments.trainer_config.trainer_config import setup_trainer
from experiments.trainer_config import trainer_args

from experiments.utils.utils import setup_log_dir

from argparse import ArgumentParser


def main(config):
    os.makedirs(config["log_path"], exist_ok=True)

    dm = setup_datamodule(config)
    task = setup_task(config)
    trainer = setup_trainer(config, trainer_args=trainer_args)
    trainer.fit(task, datamodule=dm)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--training_config", required=True)
    parser.add_argument("--cli_args", nargs="+")
    args = parser.parse_args()
    config = yaml.safe_load(open(args.training_config))
    config["cli_args"] = [arg.split(".") for arg in args.cli_args]
    log_path = setup_log_dir(config)
    config["log_path"] = log_path
    config["run_type"] = run_type = "debug" if args.debug else "experiment"
    main(config)
