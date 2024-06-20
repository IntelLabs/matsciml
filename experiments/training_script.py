import os
import yaml

from experiments.datasets.data_module_config import setup_datamodule
from experiments.task_config.task_config import setup_task
from experiments.trainer_config.trainer_config import setup_trainer

from experiments.utils.utils import setup_log_dir

from argparse import ArgumentParser


def main(config, log_path):
    # if len(args.targets) > 1:
    #     opt_target = "val.total_loss"
    # else:
    #     if args.targets[0] == "symmetry_group":
    #         opt_target = "val_spacegroup"
    #     else:
    #         opt_target = f"val_{args.targets[0]}"
    os.makedirs(log_path, exist_ok=True)

    dm = setup_datamodule(config)
    task = setup_task(config)
    trainer = setup_trainer(config)
    trainer.fit(task, datamodule=dm)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--training_config", required=True)
    args = parser.parse_args()
    config = yaml.safe_load(open(args.training_config))
    log_path = setup_log_dir(config)
    config["run_type"] = run_type = "debug" if args.debug else "experiment"
    main(config, log_path=log_path)

# from __future__ import annotations

# import os

# from argparse import ArgumentParser

# from experiments.data_config import *
# from experiments.trainer_config import *
# from experiments.training_utils import *


# do_ip_setup()


# def main(args, log_path):
#     check_args(args, data_targets)
#     if len(args.targets) > 1:
#         opt_target = "val.total_loss"
#     else:
#         if args.targets[0] == "symmetry_group":
#             opt_target = "val_spacegroup"
#         else:
#             opt_target = f"val_{args.targets[0]}"
#     os.makedirs(log_path, exist_ok=True)

#     # callbacks = setup_callbacks(opt_target, log_path)
#     # logger = setup_logger(log_path)
#     trainer_args = setup_trainer_args()

#     dm = setup_datamodule(args)
#     task = setup_task(args)
#     trainer = setup_trainer(args, callbacks, logger)
#     trainer.fit(task, datamodule=dm)

#     # trainer.model.to(device="cpu")
#     # trainer.save_checkpoint(
#     #     "/workspace/nosnap/matsciml/checkpoints/mace_sam_test_10k_may1_24.ckpt"
#     # )


# if __name__ == "__main__":
#     parser = ArgumentParser()
#     parser.add_argument("--debug", action="store_true")
#     parser.add_argument(
#         "--model",
#         required=True,
#         choices=["egnn", "megnet", "faenet", "m3gnet", "gala", "mace", "tensornet"],
#     )
#     parser.add_argument("--data", nargs="+", required=True, choices=data_keys)

#     parser.add_argument(
#         "--tasks",
#         nargs="+",
#         required=True,
#         choices=["sr", "fr", "bc", "csc", "mef", "gffr"],
#         help="ScalarRegressionTask\nForceRegressionTask\nBinaryClassificationTask\nCrystalSymmetryClassificationTask\nMaceEnergyForceTask\nGradFreeForceRegressionTask",
#     )

#     parser.add_argument(
#         "--targets",
#         nargs="+",
#         required=True,
#         default="energy",
#     )

#     parser.add_argument("--gpus", default=1, help="Number of gpu's to use")
#     parser.add_argument("--num_nodes", default=1, help="Number of nodes to use")

#     args = parser.parse_args()
#     if args.debug:
#         args.run_type = "debug"
#     else:
#         args.run_type = "experiment"

#     # log_path = os.path.join(
#     #     "/workspace/nosnap/matsciml/full-runs/",
#     #     args.run_type,
#     #     args.model,
#     #     "-".join(args.data),
#     #     "-".join(args.targets),
#     # )
#     try:
#         main(args, log_path)
#     except Exception as e:
#         error_log(e, log_path)


# # Examples

# # Single Task Single Dataset
# # python experiments/training_script.py --model megnet --data materials-project --task sr --targets energy_per_atom --gpus 1

# # MultiTask Single Dataset
# # python experiments/training_script.py --model faenet --data mp-traj --task sr gffr --targets corrected_total_energy force --gpus 1

# # Multi Data Multi Task
# # python experiments/training_script.py --model egnn --data materials-project s2ef --task sr sr --targets formation_energy_per_atom energy --gpus 1
