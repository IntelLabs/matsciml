from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Union

import pytorch_lightning as pl
import torch
from torch import nn

from matsciml.common.registry import registry
from matsciml.common.types import BatchDict, DataDict


class BaseInferenceTask(ABC, pl.LightningModule):
    def __init__(self, pretrained_model: nn.Module, *args, **kwargs):
        super().__init__()
        self.model = pretrained_model

    @abstractmethod
    def predict_step(
        self,
        batch: BatchDict,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> Any:
        ...

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
        ), f"Encoder checkpoint filepath specified but does not exist."
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
