from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any
from logging import getLogger

import pytorch_lightning as pl
import torch
from torch import nn

from matsciml.common.registry import registry
from matsciml.common.types import BatchDict, DataDict
from matsciml.models.base import BaseTaskModule, MultiTaskLitModule


class ParityData:
    def __init__(self, name: str) -> None:
        """
        Class to help accumulate inference results.

        This class should be created per target, and uses property
        setters to accumulate target and prediction tensors,
        and at the final step, aggregate them all into a single
        tensor and with the `to_json` method, produce serializable
        data.

        Parameters
        ----------
        name : str
            Name of the target property being tracked.
        """
        super().__init__()
        self.name = name
        self.logger = getLogger(f"matsciml.inference.{name}-parity")

    @property
    def ndim(self) -> int:
        if not hasattr(self, "_targets"):
            raise RuntimeError("No data set to accumulator yet.")
        sample = self._targets[0]
        if isinstance(sample, torch.Tensor):
            return sample.ndim
        else:
            return 0

    @property
    def targets(self) -> torch.Tensor:
        return torch.vstack(self._targets)

    @targets.setter
    def targets(self, values: torch.Tensor) -> None:
        if not hasattr(self, "_targets"):
            self._targets = []
        if isinstance(values, torch.Tensor):
            # remove errenous "empty" dimensions
            values.squeeze_()
        self._targets.append(values)

    @property
    def predictions(self) -> torch.Tensor:
        return torch.vstack(self._targets)

    @predictions.setter
    def predictions(self, values: torch.Tensor) -> None:
        if not hasattr(self, "_predictions"):
            self._predictions = []
        if isinstance(values, torch.Tensor):
            values.squeeze_()
        self._predictions.append(values)

    def to_json(self) -> dict[str, list]:
        return_dict = {}
        targets = self.targets.cpu()
        predictions = self.predictions.cpu()
        # do some preliminary checks to the data
        if targets.ndim != predictions.ndim:
            self.logger.warning(
                "Target/prediction dimensionality mismatch\n"
                f"  Target: {targets.ndim}, predictions: {predictions.ndim}"
            )
        if targets.shape != predictions.shape:
            self.logger.warning(
                "Target/prediction shape mismatch\n"
                f"  Target: {targets.shape}, predictions: {predictions.shape}."
            )
        return_dict["predictions"] = predictions.tolist()
        return_dict["targets"] = targets.tolist()
        return_dict["name"] = self.name
        return return_dict


class BaseInferenceTask(ABC, pl.LightningModule):
    def __init__(self, pretrained_model: nn.Module, *args, **kwargs):
        super().__init__()
        self.model = pretrained_model

    def training_step(self, *args, **kwargs) -> None:
        """Overrides Lightning method to prevent task being used for training."""
        raise NotImplementedError(
            f"{self.__class__.__name__} is not intended for training."
        )

    @abstractmethod
    def predict_step(
        self,
        batch: BatchDict,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> Any: ...

    @classmethod
    def from_pretrained_checkpoint(
        cls,
        task_ckpt_path: str | Path,
    ) -> BaseInferenceTask:
        """
        Instantiate a ``BaseInferenceTask`` from an existing Lightning checkpoint
        that contains an encoder definition.

        This requires that ``encoder_class`` and ``encoder_kwargs`` are saved as
        part of the hyperparameters.

        Parameters
        ----------
        task_ckpt_path : Union[str, Path]
            Path to an existing task checkpoint file. Typically, this
            would be a PyTorch Lightning checkpoint.

        Examples
        --------
        Load in a checkpoint, and combine it with ``InferenceWriter`` to run
        distributed inference.

        >>> from matsciml.lightning.callbacks import InferenceWriter
        >>> task = BaseInferenceTask.from_pretrained_checkpoint("epoch=0-step=100.ckpt")
        >>> trainer = pl.Trainer(callbacks=[InferenceWriter("./inference-results")], devices=2)
        >>> trainer.predict(task, datamodule=dm)
        """
        if isinstance(task_ckpt_path, str):
            task_ckpt_path = Path(task_ckpt_path)
        assert (
            task_ckpt_path.exists()
        ), "Encoder checkpoint filepath specified but does not exist."
        ckpt = torch.load(task_ckpt_path)
        select_kwargs = {}
        for key in ["encoder_class", "encoder_kwargs"]:
            assert (
                key in ckpt["hyper_parameters"]
            ), f"{key} expected to be in hyperparameters, but was not found."
            # copy over the data for the new task
            select_kwargs[key] = ckpt["hyper_parameters"][key]
        # this only copies over encoder weights, and removes the 'encoder.'
        # pattern from keys
        encoder_weights = {
            key.replace("encoder.", ""): tensor
            for key, tensor in ckpt["state_dict"].items()
            if "encoder." in key
        }
        encoder_class = select_kwargs["encoder_class"]
        encoder_kwargs = select_kwargs["encoder_kwargs"]
        # instantiate the encoder and load in pre-trained weights
        encoder = encoder_class(**encoder_kwargs)
        encoder.load_state_dict(encoder_weights)
        return cls(encoder)


@registry.register_task("EmbeddingInferenceTask")
class EmbeddingInferenceTask(BaseInferenceTask):
    def __init__(self, pretrained_model: nn.Module, *args, **kwargs):
        """
        Instantiate an ``EmbeddingInferenceTask`` with a pretrained model.

        This task simply iterates over all of the data samples, and computes
        embeddings for them based on the pretrained encoder.

        Combine this class with ``matsciml.lightning.callbacks.InferenceWriter``
        to serialize distributed inference results.
        """
        super().__init__(pretrained_model, *args, **kwargs)

    @property
    def encoder(self) -> nn.Module:
        if hasattr(self.model, "encoder"):
            return self.model.encoder
        else:
            return self.model

    def forward(self, batch: BatchDict) -> torch.Tensor:
        return self.encoder(batch)

    def predict_step(
        self,
        batch: BatchDict,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> DataDict:
        embeddings = self(batch)
        return_dict = {"embedding": embeddings}
        for key in ["targets", "symmetry"]:
            return_dict[key] = batch.get(key)
        return return_dict


@registry.register_task("ParityInferenceTask")
class ParityInferenceTask(BaseInferenceTask):
    def __init__(self, pretrained_model: BaseTaskModule):
        """
        Use a pretrained model to produce pair-plot data, i.e. predicted vs.
        ground truth.

        Example usage
        -------------
        The intended usage is to load a pretrained model, define a data module
        that points to some data to perform predictions with, then call Lightning
        Trainer's ``predict`` method.

        >>> task = ParityInferenceTask.from_pretrained_checkpoint(...)
        >>> dm = MatSciMLDataModule("DatasetName", pred_path=...)
        >>> trainer = pl.Trainer()
        >>> trainer.predict(task, datamodule=dm)

        Parameters
        ----------
        pretrained_model : BaseTaskModule
            An instance of a subclass of ``BaseTaskModule``, e.g. a
            ``ForceRegressionTask`` object.

        Raises
        ------
        NotImplementedError
            Currently, multitask modules are not yet supported.
        """
        if isinstance(pretrained_model, MultiTaskLitModule):
            raise NotImplementedError(
                "ParityInferenceTask currently only supports single task modules."
            )
        assert hasattr(pretrained_model, "predict") and callable(
            pretrained_model.predict
        ), "Model passed does not have a `predict` method; is it a `matsciml` task?"
        super().__init__(pretrained_model)
        self.common_keys = set()
        self.accumulators = {}

    def forward(self, batch: BatchDict) -> dict[str, float | torch.Tensor]:
        """
        Forward call for the inference task. This wraps the underlying
        ``matsciml`` task module's ``predict`` function to ensure that
        normalization is 'reversed', i.e. predictions are reported in
        the original unit space.

        Parameters
        ----------
        batch : BatchDict
            Batch of samples to process.

        Returns
        -------
        dict[str, float | torch.Tensor]
            Prediction output, which should correspond to a key/tensor
            mapping of output head/task name, and the associated outputs.
        """
        preds = self.model.predict(batch)
        return preds

    def on_predict_start(self) -> None:
        """Verify that logging is enabled, as it is needed."""
        if not self.trainer.log_dir:
            raise RuntimeError(
                "ParityInferenceTask requires logging to be enabled; no `log_dir` detected in Trainer."
            )

    def predict_step(
        self, batch: BatchDict, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        predictions = self(batch)
        pred_keys = set(list(predictions.keys()))
        batch_keys = set(list(batch["targets"].keys()))
        self.common_keys = pred_keys.intersection(batch_keys)
        # loop over keys that are mutually available in predictions and data
        for key in self.common_keys:
            if key not in self.accumulators:
                self.accumulators[key] = ParityData(key)
            acc = self.accumulators[key]
            acc.targets = batch["targets"][key].detach()
            acc.predictions = predictions[key].detach()

    def on_predict_epoch_end(self) -> None:
        """At the end of the dataset, write results to ``<log_dir>/inference_data.json``."""
        log_dir = Path(self.trainer.log_dir)
        output_file = log_dir.joinpath("inference_data.json")
        with open(output_file, "w+") as write_file:
            data = {key: acc.to_json() for key, acc in self.accumulators.items()}
            json.dump(data, write_file, indent=2)
