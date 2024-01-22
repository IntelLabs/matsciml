from __future__ import annotations

from argparse import ArgumentParser
from copy import deepcopy
from pathlib import Path

import pytorch_lightning as pl
from munch import Munch, munchify, unmunchify
from pytorch_lightning.loggers import CSVLogger
from tqdm import tqdm
from yaml import safe_load

from matsciml import datasets as dsets
from matsciml.datasets.transforms import DistancesTransform, GraphVariablesTransform
from matsciml.lightning.callbacks import ForwardNaNDetection, LeaderboardWriter
from matsciml.lightning.cli import DATAMODULE_REGISTRY, MODEL_REGISTRY
from matsciml.lightning.data_utils import IS2REDGLDataModule, is2re_devset
from matsciml.models import GraphConvModel, IS2RELitModule

# import the callback responsible for aggregating and formatting
# prediction results


def verify_input_args(args):
    # make sure all paths are valid
    for key in ["backbone_config", "ckpt_path", "data_config"]:
        assert Path(
            getattr(args, key),
        ).exists(), f"{key} does not correspond to a valid path! {getattr(args, key)}"

    for val_path in args.val_paths:
        assert Path(val_path).exists(), f"{val_path} does not exist!"

    # make sure backbone, task, data references are valid
    assert (
        args.backbone in MODEL_REGISTRY
    ), f"{args.backbone} is not a valid backbone: {MODEL_REGISTRY}"
    # get the task lightning modules
    lit_modules = [
        v
        for v in MODEL_REGISTRY
        if any([key in v for key in ["S2EF", "IS2RE", "PointCloud"]])
    ]
    assert (
        args.task in MODEL_REGISTRY
    ), f"{args.task} is not a valid task lightning module: {lit_modules}"
    assert args.task_data_module in DATAMODULE_REGISTRY


def load_model(args):
    # load in configs
    with open(args.backbone_config) as read_file:
        backbone_yml = safe_load(read_file)
    # configure backbone
    backbone_cls = MODEL_REGISTRY.get(args.backbone)
    backbone = backbone_cls(**backbone_yml["model"]["init_args"]["gnn"]["init_args"])
    model_cls = MODEL_REGISTRY.get(args.task)
    model = model_cls(**backbone_yml["model"]["init_args"]).load_from_checkpoint(
        gnn=backbone,
        checkpoint_path=args.ckpt_path,
    )
    return model


def load_datamodule(args, path):
    with open(args.data_config) as read_file:
        data_yml = safe_load(read_file)

    # match the right data module
    datamodule_cls = DATAMODULE_REGISTRY.get(args.task_data_module)

    # now enter loop over all validation files

    data_kwargs = deepcopy(data_yml["data"]["init_args"])
    # patch it slightly to specialize for point clouds, which requires wrapping the dataset
    if "PointCloud" in args.task_data_module:
        dataset_classname = data_kwargs.get("dataset_class").split(".")[-1]
        dset_class = getattr(dsets, dataset_classname)
        data_kwargs["dataset_class"] = dset_class

    # set the correct validation path
    if args.pipeline.lower() == "predict":
        data_kwargs["predict_path"] = path
    elif args.pipeline.lower() == "validate":
        data_kwargs["val_path"] = path

    # data_kwargs["batch_size"] = 1

    if args.backbone == "MEGNet":
        datamodule = datamodule_cls(
            transforms=[
                DistancesTransform(),
                GraphVariablesTransform(),
            ],
            **data_kwargs,
        )
    else:
        datamodule = datamodule_cls(**data_kwargs)

    return datamodule


def load_logger_and_callbacks(path):
    val_set_name = Path(path).stem
    logger = CSVLogger(
        save_dir=f"{args.logger.init_args.save_dir}_{val_set_name}",
        name=f"{args.backbone}-{args.task}-{args.logger.init_args.name}",
        version=val_set_name,
    )
    npz_log = logger.save_dir + "/" + logger.name
    callbacks = LeaderboardWriter(npz_log)
    return logger, callbacks


def load_trainer(args, logger, callbacks):
    trainer = pl.Trainer(
        accelerator=args.accelerator,
        # strategy="ddp" if args.num_devices > 1 else None,
        strategy=None,
        devices=args.num_devices,
        logger=[logger],
        callbacks=callbacks,
    )
    return trainer


def main(trainer, model, datamodule, pipeline):
    if pipeline.lower() == "predict":
        trainer.predict(model, datamodule=datamodule)
    elif pipeline.lower() == "validate":
        trainer.validate(model, datamodule=datamodule)


if __name__ == "__main__":
    """
    General purpose validation module.

    Given a variable list of validation datasets, this script will load in
    a model/backbone, and crank through each one, reporting the results in
    `validation_runs/{model}-{task}/{validation set}.
    """

    argparser = ArgumentParser()
    argparser.add_argument(
        "--config-file",
        type=str,
        default="../pl-configs/predict.yml",
        help="Configuration file for the chosen run",
    )

    cmd_args = argparser.parse_args()

    with open(cmd_args.config_file) as f:
        data_map = safe_load(f)

    args = munchify(data_map).predict
    verify_input_args(args)
    model = load_model(args)
    for path in tqdm(args.val_paths):
        datamodule = load_datamodule(args, path)
        logger, callbacks = load_logger_and_callbacks(path)
        trainer = load_trainer(args, logger, callbacks)
        main(trainer, model, datamodule, args.pipeline)
