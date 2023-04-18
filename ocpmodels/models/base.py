# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: MIT License

from typing import (
    Dict,
    Iterable,
    Type,
    Tuple,
    Optional,
    Union,
    ContextManager,
    List,
    Any,
)
from abc import abstractmethod
from contextlib import nullcontext, ExitStack
import logging
from dgl.utils import data

import pytorch_lightning as pl
import torch
from torch import Tensor, nn
import dgl
from torch.optim import AdamW, Optimizer

from ocpmodels.modules.normalizer import Normalizer
from ocpmodels.models.common import OutputHead

"""
base.py

This module implements all the base classes for task and model
abstraction.

The way models and tasks are meant to be composed is as follows:
An abstract GNN architecture inherits from either `AbstractS2EFModel`
or `AbstractIS2REModel`: this abstracts out things like force computation
in the former, where the `forward` pass computes the energy, and the
class implements the `compute_force` method that uses autograd for
the force.

The GNN model is then passed as "the model" within a PyTorch Lightning
Module, which takes care of all the loss computation, normalization,
logging, and CPU/GPU/TPU transfers.

"""


def decorate_color(color: str):
    """This creates a logging function with flair"""

    def debug_message(logger, message: str) -> None:
        logger.debug(f"\033{color} {message}\033[00m")

    return debug_message


# set up different colors for logging
debug_green = decorate_color("[92m")
debug_lightpurple = decorate_color("[94m")
debug_cyan = decorate_color("[96m")


def dynamic_gradients_context(need_grad: bool, has_rnn: bool) -> ContextManager:
    """
    Conditional gradient context manager, based on whether or not
    force computation is necessary in the process.
    This is necessary because there are actually two contexts
    necessary: enable gradient computation _and_ make sure we
    aren't in inference mode, which is enabled by PyTorch Lightning
    for faster inference.
    If this is `regress_forces` is set to False, a `nullcontext`
    is applied that does nothing.
    Parameters
    ----------
    need_grad : bool
        Flag to designate whether or not gradients need to be forced
        within this code block.
    has_rnn : bool
        Flag to indicate whether or not RNNs are being used in this
        model, which will disable cudnn to enable double backprop.
    Returns
    -------
    ContextManager
        Joint context, combining `inference_mode` and `enable_grad`,
        otherwise a `nullcontext` if `need_grad` is `False`.
    """
    manager = ExitStack()
    if need_grad:
        contexts = [torch.inference_mode(False), torch.enable_grad()]
        # if we're also using CUDA, there is an additional context to allow
        # RNNs to do double backprop
        if torch.cuda.is_available() and has_rnn:
            contexts.append(torch.backends.cudnn.flags(enabled=False))
        for cxt in contexts:
            manager.enter_context(cxt)
    else:
        manager.enter_context(nullcontext())
    return manager


def rnn_force_train_mode(module: nn.Module) -> None:
    """
    Forces RNN subclasses into training mode to facilitate
    derivatives for force computation outside of training
    steps.
    See https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnRNNForward
    Parameters
    ----------
    module : nn.Module
        Abstract `torch.nn.Module` to check and toggle
    """
    # this try/except will catch non-CUDA enabled systems
    # this patch is only for cudnn
    try:
        _ = torch.cuda.current_device()
        if isinstance(module, nn.RNNBase):
            module.train()
    except RuntimeError:
        pass


def lit_conditional_grad(regress_forces: bool):
    """
    Decorator function that will dynamically enable gradient
    computation. An example usage for this decorator is given in
    the `S2EFLitModule.forward` call, where we determine at
    runtime whether or not to enable gradients for the force
    computation by wrapping the embedded `gnn.forward` method.

    Parameters
    ----------
    regress_forces : bool
        Specifies whether or not to regress forces; if so,
        enable gradient computation.
    """

    def decorator(func):
        def cls_method(self, *args, **kwargs):
            f = func
            if regress_forces:
                f = torch.enable_grad()(func)
            return f(self, *args, **kwargs)

        return cls_method

    return decorator


def prepend_affix(metrics: Dict[str, torch.Tensor], affix: str) -> None:
    """
    Mutate a dictionary in place, prepending an affix to keys.

    This is primarily for logging metrics, where we want to denote something
    originating from train/test/validation, etc.

    Parameters
    ----------
    metrics : Dict[str, torch.Tensor]
        Dictionary containing metrics
    affix : str
        Affix to prepend each key, for example "train" for training metrics.
    """
    keys = list(metrics.keys())
    for key in keys:
        metrics[f"{affix}_{key}"] = metrics[key]
        del metrics[key]


class BaseModel(nn.Module):
    def __init__(self, num_atoms=None, bond_feat_dim=None, num_targets=None):
        super(BaseModel, self).__init__()
        self.num_atoms = num_atoms
        self.bond_feat_dim = bond_feat_dim
        self.num_targets = num_targets

    def forward(self, data):
        raise NotImplementedError

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())


class AbstractTask(pl.LightningModule):

    __task__ = None

    def __init__(self) -> None:
        super().__init__()

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class OCPLitModule(pl.LightningModule):

    __normalize_keys__ = ["target", "grad_target"]

    def __init__(
        self,
        gnn: AbstractTask,
        normalize_kwargs: Optional[Dict[str, float]] = None,
        nan_check: bool = False,
    ):
        super().__init__()
        # TODO phase out `gnn` as the variable name, as we don't only
        # work with GNNs
        self.gnn = gnn
        self.model = self.gnn
        self.normalizers = {}
        self._nan_check = nan_check
        for key in self.__normalize_keys__:
            # if no values are provided in the dictionary, the scaling
            # should have no effect
            if isinstance(normalize_kwargs, dict):
                mean, std = (
                    normalize_kwargs.get(f"{key}_mean", 0.0),
                    normalize_kwargs.get(f"{key}_std", 1),
                )
            else:
                mean, std = 0.0, 1.0
            self.normalizers[key] = Normalizer(mean=mean, std=std)
        if self._nan_check:
            # configure logging for the bad batch detection
            self._nan_logger = logging.getLogger("pytorch_lightning")
            self._nan_logger.setLevel(logging.DEBUG)
            self._nan_logger.addHandler(logging.FileHandler("nan_checker.log"))

    def forward(self, *args, **kwargs):
        """
        Wrap the LitModule forward pass as whatever the abstract underlying
        model uses.

        Returns
        -------
        _type_
            _description_
        """
        return self.gnn(*args, **kwargs)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, self.hparams.gamma
        )
        return [optimizer], [lr_scheduler]

    @property
    def has_rnn(self) -> bool:
        """
        Property to determine whether or not this LightningModule contains
        RNNs. This is primarily to determine whether or not to enable/disable
        contexts with cudnn, as double backprop is not supported.
        Returns
        -------
        bool
            True if any module is a subclass of `RNNBase`, otherwise False.
        """
        return any([isinstance(module, nn.RNNBase) for module in self.modules()])

    def _nan_check_gradients(self, batch_idx: int) -> bool:
        """
        Check model parameters for NaNs prior to backprop. Will return
        True if there are any gradients that are NaN, which will be
        used to skip the gradient update for this batch.

        TODO make this a PyTorch Lightning integration/callback


        Parameters
        ----------
        batch_idx : int
            Batch index to debug with

        Returns
        -------
        bool
            True if there are NaN gradients, otherwise False
        """
        for param in self.gnn.parameters():
            if param.requires_grad and param.grad is not None:
                if torch.any(torch.isnan(param.grad)):
                    debug_green(
                        self._nan_logger,
                        "Exploding Gradient with NaN; Batch IDX: {}".format(batch_idx),
                    )
                    return True
        return False

    def step(
        self,
        batch: Dict[str, Union[torch.Tensor, dgl.DGLGraph]],
        batch_idx: int,
        prefix: str = "validation",
    ) -> float:
        """self, batch: Dict[str, Union[torch.Tensor, dgl.DGLGraph]], batch_idx: int
        ) -> Dict[str, Union[float, Dict[str, float]]]:
            Step function for an abstract part of the train/val/test pipeline.
            If the specific functions are not overwritten, the default behavior
            is to just compute the loss metrics and log them.

            Parameters
            ----------
            batch : Dict[str, Union[torch.Tensor, dgl.DGLGraph]]
                A dictionary of batched data, including one "graph" key
                used as an input DGL graph.
            batch_idx : int
                Index of the batch being processed.
            prefix : str, optional
                String prefix for metric logging, by default "validation"

            Returns
            -------
            float
                The value for the summed loss
        """
        losses = self._compute_losses(batch, batch_idx)
        # ensure batch size is correct
        batch_size = self._get_batch_size(batch)
        for key, value in losses.items():
            self.log(
                f"{prefix}_{key}",
                value,
                on_step=True,
                batch_size=batch_size,
                prog_bar=True,
            )
        return losses.get("loss")

    @abstractmethod
    def training_step(
        self, batch: Dict[str, Union[torch.Tensor, dgl.DGLGraph]], batch_idx: int
    ) -> Dict[str, Union[float, Dict[str, float]]]:
        """
        This function implements the logic for a training step of one of
        the task pipelines. Tasks that subclass `OCPLitModule` must
        implement this function in addition to `_compute_losses` to
        describe the minimal behavior for a task.

        Parameters
        ----------
        batch : Dict[str, Union[torch.Tensor, dgl.DGLGraph]]
            Dictionary of batched data
        batch_idx : int
            Batch index

        Returns
        -------
        Dict[str, Union[float, Dict[str, float]]]
            Nested dictionary of losses, including the sum (float "loss") and
            all individual components (dictionary "logs")
        """
        raise NotImplementedError

    def validation_step(
        self, batch: Dict[str, Union[torch.Tensor, dgl.DGLGraph]], batch_idx: int
    ) -> float:
        """
        Implements the default behavior for validation, which is to
        just compute the same loss metrics as for training, however
        uses "validation" for the logging prefix.

        Parameters
        ----------
        batch : Dict[str, Union[torch.Tensor, dgl.DGLGraph]]
            A dictionary of batched data, including one "graph" key
            used as an input DGL graph.
        batch_idx : int
            Index of the batch being processed.

        Returns
        -------
        float
            Value for the summed loss
        """
        return self.step(batch, batch_idx, "validation")

    def test_step(
        self, batch: Dict[str, Union[torch.Tensor, dgl.DGLGraph]], batch_idx: int
    ) -> float:
        return self.step(batch, batch_idx, "test")

    def predict_step(
        self,
        batch: Dict[str, Union[torch.Tensor, dgl.DGLGraph]],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> Dict[str, torch.Tensor]:
        """
        Implements the inference logic for energy prediction, which is used to
        run the leaderboard oriented prediction pipeline.

        This is intended to be used in tandem with the `LeaderboardWriter` callback,
        which will save and format the results from inference in a way that conforms
        with the evalAI formatting.

        Parameters
        ----------
        batch : Dict[str, Union[torch.Tensor, dgl.DGLGraph]]
            Batch of data, corresponding to a dictionary of tensors
        batch_idx : int
            Index of the batch
        dataloader_idx : int, optional
            Dataloader index, which is used for DDP processing, by default 0

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary containing the IDs of each batch item, and the corresponding
            energy result.
        """
        input_data = self._get_inputs(batch)
        prediction = self(*input_data)
        normalizer = self.normalizers.get("target", None)
        if normalizer is not None:
            prediction = normalizer.denorm(prediction)
        ids = batch.get("sid")
        return {"id": ids, "energy": prediction}

    @abstractmethod
    def _compute_losses(
        self, batch: Dict[str, Union[torch.Tensor, dgl.DGLGraph]], batch_idx: int
    ) -> Dict[str, Union[float, Dict[str, float]]]:
        """
        Compute the various loss terms, given a batch, and return
        key/value mapping of the type of loss and its value. At
        a bare minimum, you _must_ return a dictionary with a "loss"
        key, which represents the sum of all losses.

        Parameters
        ----------
        batch : Dict[str, Union[torch.Tensor, dgl.DGLGraph]]
            Dictionary of batched data; expects one key "graph" corresponding
            to the input DGL graph.
        batch_idx : int
            Index for the batch.

        Returns
        -------
        Dict[str, Union[float, Dict[str, float]]]
            Nested dictionary containing the summed loss ("loss"),
            and all other metrics ("logs")
        """
        raise NotImplementedError

    def _get_inputs(
        self, batch: Dict[str, Union[torch.Tensor, dgl.DGLGraph]]
    ) -> Tuple[Union[torch.Tensor, dgl.DGLGraph]]:
        """
        Defines a method for extracting inputs out of a batch,
        to be readily unpacked into a forward method for the
        model.

        The idea behind this method is to allow changes in the
        data representation without needing to re-implement
        the step/compute_losses functions.

        Parameters
        ----------
        batch : Dict[str, Union[torch.Tensor, dgl.DGLGraph]]
            Dictionary containing batched data

        Returns
        -------
        Tuple[Union[torch.Tensor, dgl.DGLGraph]]
            Tuple of input data to the model, which is
            subsequently unpacked into `forward`.
        """
        graph = batch.get("graph")
        return (graph,)

    def _get_batch_size(
        self, batch: Dict[str, Union[torch.Tensor, dgl.DGLGraph]]
    ) -> int:
        """
        Defines a method for getting the batch size from a batch.

        Requires the user to define a reliable method of evaluating
        the batch size from one of the items contained in `batch.

        Parameters
        ----------
        batch : Dict[str, Union[torch.Tensor, dgl.DGLGraph]]
            Dictionary of batched data

        Returns
        -------
        int
            Batch size
        """
        return batch.get("graph").batch_size


class IS2RELitModule(OCPLitModule):
    """
    Implements a class for the IS2RE task of OCP; initial structure to
    relaxed energy prediction.

    This overrides the `_compute_losses` method with a concrete one, whereby
    the loss is referred here as an autoencoding loss (i.e. given an unoptimized
    molecular graph, return a "relaxed" one).
    """

    def __init__(
        self,
        gnn: AbstractTask,
        lr: float,
        gamma: float,
        energy_coefficient: float = 1.0,
        energy_loss: Optional[Type[nn.Module]] = nn.L1Loss,
        normalize_kwargs: Optional[Dict[str, float]] = None,
        nan_check: bool = False,
    ):
        super().__init__(gnn, normalize_kwargs, nan_check)
        self.lr = lr
        self.gamma = gamma
        self.scalers = {
            "energy": energy_coefficient,
        }
        if not isinstance(energy_loss, nn.Module):
            energy_loss = energy_loss()
        self.energy_loss = energy_loss
        self.save_hyperparameters(ignore=["gnn", "energy_loss"])

    def _compute_losses(
        self, batch: Dict[str, Union[torch.Tensor, dgl.DGLGraph]], batch_idx: int
    ) -> Dict[str, Union[float, Dict[str, float]]]:
        """
        Compute the loss for the S2EF task.

        The code here implements the high level task, whereby an abstract
        GNN takes in a graph and predicts its electronic energy. If we are
        also predicting the force, we will also unpack the first derivative
        of energy w.r.t. atomic positions.

        Parameters
        ----------
        batch : Dict[str, Union[torch.Tensor, dgl.DGLGraph]]
            A dictionary of batched data from the S2EF dataset
        batch_idx : int
            Index of the batch being worked on

        Returns
        -------
        Dict[str, Union[float, Dict[str, float]]]
            Nested dictionary, with one top level key "loss" used
            by PyTorch Lightning
        """
        input_data = self._get_inputs(batch)
        true_energies = batch.get("y_relaxed")
        pred_energy = self(*input_data).squeeze()
        # normalize the targets before loss computation
        true_energies = self.normalizers["target"].norm(true_energies)
        # compute energy and force losses
        energy_loss = self.energy_loss(pred_energy, true_energies)
        if self._nan_check:
            if energy_loss != energy_loss:
                debug_cyan(
                    self._nan_logger,
                    f"Bad Batch with NaN F-Prop; batch index {batch_idx}",
                )
            # trigger the NaN checking in the parameter gradients; skip
            # the batch if there are bad gradients
            nan_grad_check = self._nan_check_gradients(batch_idx)
            if nan_grad_check:
                return None
        # package the losses into a dictionary for logging
        loss_dict = {"energy": energy_loss}
        # get coefficients to rescale the losses
        for key in loss_dict.keys():
            loss_dict[key] *= self.scalers.get(key)
        loss_dict["total"] = sum([value for value in loss_dict.values()])
        # this is somewhat unecessary, but makes it consistent with other tasks
        return {"loss": loss_dict["total"], "logs": loss_dict}

    def training_step(
        self, batch: Dict[str, Union[torch.Tensor, dgl.DGLGraph]], batch_idx: int
    ) -> Dict[str, Union[float, Dict[str, float]]]:
        return self.step(batch, batch_idx, prefix="train")


class IS2REPointCloudModule(IS2RELitModule):
    def forward(
        self, features: torch.Tensor, positions: torch.Tensor, *args, **kwargs
    ) -> torch.Tensor:
        """
        Implements the forward method for a point cloud representation.

        Instead of passing a DGL graph, the model should expect `features`
        and `positions` args; additional args and kwargs are passed into
        the abstract model's `forward` method.

        Parameters
        ----------
        features : torch.Tensor
            N-D Tensor containing features, with shape [B, *] for B batch
            entries.
        positions : torch.Tensor
            Tensor containing atom positions

        Returns
        -------
        torch.Tensor
            Float Tensor containing the energy of each batch entry
        """
        return self.model(features, positions, *args, **kwargs)

    def _get_inputs(
        self, batch: Dict[str, Union[torch.Tensor, dgl.DGLGraph]]
    ) -> Tuple[Union[torch.Tensor, dgl.DGLGraph]]:
        """Get the input data from keys `pc_features` and `pos`"""
        features, positions = batch.get("pc_features"), batch.get("pos")
        return (features, positions)

    def _get_batch_size(
        self, batch: Dict[str, Union[torch.Tensor, dgl.DGLGraph]]
    ) -> int:
        """Determine the batch size from the first dimension of `pc_features`"""
        return int(batch.get("pc_features").size(0))


class S2EFLitModule(OCPLitModule):
    """
    Losses:

    Loss calculation relies on the class attributes `energy_loss`
    and `force_loss`, which in turn is expected to refer to an `nn.Module`.
    The idea behind this is to allow the user to compose their own
    loss metrics in the YAML config, or by default, just rely on the
    `L1Loss`.

    The hyperparameters `energy_coefficient` and `force_coefficient` are
    also used to scale the respective losses.
    """

    def __init__(
        self,
        gnn: AbstractTask,
        lr: float,
        gamma: float,
        energy_coefficient: float = 1.0,
        force_coefficient: float = 1.0,
        energy_loss: Optional[Type[nn.Module]] = nn.L1Loss,
        force_loss: Optional[Type[nn.Module]] = nn.L1Loss,
        normalize_kwargs: Optional[Dict[str, float]] = None,
        regress_forces: Optional[bool] = True,
        nan_check: bool = False,
    ):
        super().__init__(gnn, normalize_kwargs, nan_check)
        self.lr = lr
        self.gamma = gamma
        self.scalers = {
            "energy": energy_coefficient,
            "force": force_coefficient,
        }
        # TODO check that this instantiates correctly in the CLI
        # instantiate the classes, if they aren't already
        if not isinstance(energy_loss, nn.Module):
            energy_loss = energy_loss()
        if not isinstance(force_loss, nn.Module):
            force_loss = force_loss()
        self.energy_loss = energy_loss
        self.force_loss = force_loss
        self.regress_forces = regress_forces
        self.save_hyperparameters(ignore=["gnn", "energy_loss", "force_loss"])
        # this lets us manually override the optimization process
        # and backward ourselves; necessary for force derivatives
        self.automatic_optimization = False

    def forward(
        self, graph: dgl.DGLGraph, *args, **kwargs
    ) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
        """
        Use the embedded GNN to compute the energy, and if `regress_forces` is
        True, compute the force (as the derivative of energy w.r.t. atomic
        positions) as well.

        Additional arguments and kwargs are passed to the GNN forward method,
        but at the bare minimum requires you to pass a `DGLGraph` object.

        Parameters
        ----------
        graph : dgl.DGLGraph
            DGLGraph object

        Returns
        -------
        Union[torch.Tensor, Tuple[torch.Tensor]]
            If `regress_forces` is True, return a 2-tuple of energy, force.
            Otherwise, just return the energy.
        """
        if self.regress_forces:
            # make sure atomic positions are tracking gradients
            # for the force computation
            graph.ndata["pos"].requires_grad_(True)
        # decorate the GNN's forward method, which will enable/disable
        # gradient computation as necessary
        compute_func = lit_conditional_grad(self.regress_forces)(self.gnn.forward)
        energy = compute_func(graph, *args, **kwargs)
        if self.regress_forces:
            forces = (
                -1
                * torch.autograd.grad(
                    energy,
                    graph.ndata["pos"],
                    grad_outputs=torch.ones_like(energy),
                    create_graph=True,
                )[0]
            )
            return (energy, forces)
        return energy

    def training_step(
        self, batch: Dict[str, Union[torch.Tensor, dgl.DGLGraph]], batch_idx: int
    ) -> Dict[str, Union[float, Dict[str, float]]]:
        """
        Implements the training logic for S2EF.

        Shares the `_compute_losses` method as with the other steps, however
        because we are doing a double-gradient computation for training and
        for the force computation, we have to do so manually and out of the
        regular abstraction for PyTorch Lightning.

        This will also automatically log the metrics into the trainer's logger.

        Parameters
        ----------
        batch : Dict[str, Union[torch.Tensor, dgl.DGLGraph]]
            A dictionary of batched data from the S2EF dataset.
        batch_idx : int
            Index of the batch being processed.

        Returns
        -------
        Dict[str, Union[float, Dict[str, float]]]
            Nested dictionary of losses
        """
        # this forces gradient computation
        with dynamic_gradients_context(self.regress_forces, self.has_rnn):
            # grab the single optimizer
            optimizer = self.optimizers()
            optimizer.zero_grad()
            # compute losses, log them, and grab the total loss for backprop
            losses = self._compute_losses(batch, batch_idx)
        batch_size = self._get_batch_size(batch)
        # log the losses individually with the batch size specified
        for key, value in losses.get("logs").items():
            self.log(
                f"train_{key}",
                value,
                on_step=True,
                batch_size=batch_size,
                prog_bar=True,
            )
        total_loss = losses.get("loss")

        if self._nan_check:
            if total_loss != total_loss:
                debug_cyan(
                    self._nan_logger,
                    f"Bad Batch with NaN F-Prop; batch index {batch_idx}",
                )

        if self.regress_forces:
            # run it back; retain_graph allows higher order derivatives
            self.manual_backward(total_loss, retain_graph=True)
        self.manual_backward(total_loss)

        # trigger the NaN checking in the parameter gradients; skip
        # the batch if there are bad gradients
        if self._nan_check:
            nan_grad_check = self._nan_check_gradients(batch_idx)
            if nan_grad_check:
                return None

        optimizer.step()
        return losses

    def _compute_losses(
        self, batch: Dict[str, Union[torch.Tensor, dgl.DGLGraph]], batch_idx: int
    ) -> Dict[str, Union[float, Dict[str, float]]]:
        """
        Compute the loss for the S2EF task.

        The code here implements the high level task, whereby an abstract
        GNN takes in a graph and predicts its electronic energy. If we are
        also predicting the force, we will also unpack the first derivative
        of energy w.r.t. atomic positions.

        Parameters
        ----------
        batch : Dict[str, Union[torch.Tensor, dgl.DGLGraph]]
            A dictionary of batched data from the S2EF dataset
        batch_idx : int
            Index of the batch being worked on

        Returns
        -------
        Dict[str, Union[float, Dict[str, float]]]
            Nested dictionary, with one top level key "loss" used
            by PyTorch Lightning
        """
        inputs = self._get_inputs(batch)
        if "graph" in batch:
            true_forces = inputs[0].ndata["force"]
        else:
            true_forces = batch.get("force")
        true_energies = batch.get("y")

        if self.regress_forces:
            (pred_energy, pred_force) = self(*inputs)
        else:
            pred_energy = self(*inputs)
        # normalize the targets before loss computation
        true_energies = self.normalizers["target"].norm(true_energies)
        true_forces = self.normalizers["grad_target"].norm(true_forces)
        # compute energy and force losses
        energy_loss = self.energy_loss(pred_energy.squeeze(), true_energies)
        # package the losses into a dictionary for logging
        loss_dict = {"energy": energy_loss}
        if self.regress_forces:
            force_loss = self.force_loss(pred_force, true_forces)
            loss_dict["force"] = force_loss
        # get coefficients to rescale the losses
        for key in loss_dict.keys():
            loss_dict[key] *= self.scalers.get(key)
        loss_dict["total"] = sum([value for value in loss_dict.values()])
        return {"loss": loss_dict["total"], "logs": loss_dict}

    def on_validation_start(self) -> None:
        self.apply(rnn_force_train_mode)

    def on_test_start(self) -> None:
        self.apply(rnn_force_train_mode)

    def on_predict_start(self) -> None:
        self.apply(rnn_force_train_mode)

    def on_validation_batch_end(
        self, outputs: Any, batch: Any, batch_idx: int, dataloader_idx: int
    ) -> None:
        super().on_predict_batch_end(outputs, batch, batch_idx, dataloader_idx)
        # ensure gradients aren't contaminating any results between batches
        self.zero_grad()

    def on_test_batch_end(
        self, outputs: Any, batch: Any, batch_idx: int, dataloader_idx: int
    ) -> None:
        super().on_predict_batch_end(outputs, batch, batch_idx, dataloader_idx)
        # ensure gradients aren't contaminating any results between batches
        self.zero_grad()

    def on_predict_batch_end(
        self, outputs: Any, batch: Any, batch_idx: int, dataloader_idx: int
    ) -> None:
        super().on_predict_batch_end(outputs, batch, batch_idx, dataloader_idx)
        # ensure gradients aren't contaminating any results between batches
        self.zero_grad()

    def predict_step(
        self,
        batch: Dict[str, Union[torch.Tensor, dgl.DGLGraph]],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> Dict[str, torch.Tensor]:
        # force gradients when running predictions
        with dynamic_gradients_context(
            self.regress_forces, self.has_rnn
        ) as grad_content:
            input_data = self._get_inputs(batch)
            if self.regress_forces:
                (pred_energy, pred_force) = self(*input_data)
                # detach from the graph
                pred_energy, pred_force = pred_energy.detach(), pred_force.detach()
            else:
                pred_energy = self(*input_data)
                # detach from the graph
                pred_energy = pred_energy.detach()
        ids, chunk_ids = batch.get("sid"), batch.get("fid")
        # ids are formatted differently for force tasks
        system_ids = [f"{i}_{j}" for i, j in zip(ids, chunk_ids)]
        predictions = {
            "ids": system_ids,
            "chunk_ids": chunk_ids,
            "energy": pred_energy.to(torch.float16),
        }
        # processing the forces is a bit more complicated because apparently
        # only the free atoms are considered
        if self.regress_forces:
            graph = batch.get("graph")
            fixed_mask = graph.ndata["fixed"] == 0
            # retrieve only forces corresponding to unfixed nodes
            predictions["forces"] = pred_force[fixed_mask]
            natoms = tuple(batch.get("natoms").cpu().numpy().astype(int))
            chunk_split = torch.split(graph.ndata["fixed"], natoms)
            chunk_ids = []
            for chunk in chunk_split:
                ids = (len(chunk) - sum(chunk)).cpu().numpy().astype(int)
                chunk_ids.append(int(ids))

            predictions["chunk_ids"] = chunk_ids
        return predictions


class S2EFPointCloudModule(S2EFLitModule):
    def forward(
        self, features: torch.Tensor, positions: torch.Tensor, *args, **kwargs
    ) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
        """
        Use the embedded point cloud model to compute the energy, and if `regress_forces` is
        True, compute the force (as the derivative of energy w.r.t. atomic
        positions) as well.

        Parameters
        ----------
        features : torch.Tensor
            N-D tensor containing features of the point cloud
        positions : torch.Tensor
            N-D tensor containing atom/point positions

        Returns
        -------
        Union[torch.Tensor, Tuple[torch.Tensor]]
            If `regress_forces` is True, return a 2-tuple of energy, force.
            Otherwise, just return the energy.
        """
        if self.regress_forces:
            # make sure atomic positions are tracking gradients
            # for the force computation
            positions.requires_grad_(True)
        # decorate the GNN's forward method, which will enable/disable
        # gradient computation as necessary
        compute_func = lit_conditional_grad(self.regress_forces)(self.model.forward)
        energy = compute_func(features, positions, *args, **kwargs)
        if self.regress_forces:
            forces = (
                -1
                * torch.autograd.grad(
                    energy,
                    positions,
                    grad_outputs=torch.ones_like(energy),
                    create_graph=True,
                )[0]
            )
            return (energy, forces)
        return energy

    def _get_inputs(
        self, batch: Dict[str, Union[torch.Tensor, dgl.DGLGraph]]
    ) -> Tuple[Union[torch.Tensor, dgl.DGLGraph]]:
        """Get the input data from keys `pc_features` and `pos`"""
        features, positions = batch.get("pc_features"), batch.get("pos")
        return (features, positions)

    def _get_batch_size(
        self, batch: Dict[str, Union[torch.Tensor, dgl.DGLGraph]]
    ) -> int:
        """Determine the batch size from the first dimension of `pc_features`"""
        return int(batch.get("pc_features").size(0))


class AbstractEnergyModel(AbstractTask):

    __task__ = "S2EF"

    """
    At a minimum, the point of this is to help register associated models
    with PyTorch Lightning ModelRegistry; the expectation is that you get
    the graph energy as well as the atom forces.
    """

    def __init__(self):
        super().__init__()

    def forward(self, graph: dgl.DGLGraph) -> Tensor:
        """
        Implements the basic forward call for an S2EF task; given a graph,
        predict the energy. Force computation relies on a decorated version
        of this function, which is used by the `S2EFLitModule`.

        Parameters
        ----------
        graph : dgl.DGLGraph
            A DGL graph object

        Returns
        -------
        Tensor
            A float Tensor containing the energy of
            each graph, shape [G, 1] for G graphs
        """
        energy = self.forward(graph)
        return energy


class BaseTaskModule(pl.LightningModule):

    __task__ = None

    def __init__(
        self,
        encoder: nn.Module,
        loss_func: Union[Type[nn.Module], nn.Module],
        task_keys: Optional[List[str]] = None,
        output_kwargs: Dict[str, Any] = {},
        lr: float = 1e-4,
        weight_decay: float = 0.0,
        normalize_kwargs: Optional[Dict[str, float]] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.task_keys = task_keys
        if isinstance(loss_func, Type):
            loss_func = loss_func()
        self.loss_func = loss_func
        default_heads = {"act_last": None, "hidden_dim": 128}
        default_heads.update(output_kwargs)
        self.output_kwargs = default_heads
        self.normalize_kwargs = normalize_kwargs
        self.save_hyperparameters(ignore=["encoder", "loss_func"])

    @property
    def task_keys(self) -> List[str]:
        return self._task_keys

    @task_keys.setter
    def task_keys(self, values: Union[set, List[str], None]) -> None:
        """
        Ensures that the task keys are unique.

        Parameters
        ----------
        values : Union[set, List[str]]
            Array of keys to use to look up targets.
        """
        if values is None:
            values = []
        if isinstance(values, list):
            values = set(values)
        if isinstance(values, set):
            values = list(values)
        self._task_keys = values

    @abstractmethod
    def _make_output_heads(self) -> nn.ModuleDict:
        ...

    @property
    def output_heads(self) -> nn.ModuleDict:
        return self._output_heads

    @output_heads.setter
    def output_heads(self, heads: nn.ModuleDict) -> None:
        assert isinstance(
            heads, nn.ModuleDict
        ), f"Output heads must be an instance of `nn.ModuleDict`."
        assert len(heads) > 0, f"No output heads in {heads}."
        assert all(
            [key in self.task_keys for key in heads.keys()]
        ), f"Output head keys {heads.keys()} do not match any in tasks: {self.task_keys}."
        self._output_heads = heads

    @property
    def num_heads(self) -> int:
        return len(self.task_keys)

    def forward(
        self,
        batch: Dict[str, Union[torch.Tensor, dgl.DGLGraph, Dict[str, torch.Tensor]]],
    ) -> Dict[str, torch.Tensor]:
        embedding = self.encoder(batch)
        outputs = {}
        # process each head sequentially
        for key, head in self.output_heads.items():
            outputs[key] = head(embedding)
        return outputs

    def _get_targets(
        self,
        batch: Dict[str, Union[torch.Tensor, dgl.DGLGraph, Dict[str, torch.Tensor]]],
    ) -> Dict[str, torch.Tensor]:
        """
        Method for extracting targets out of a batch.

        Ultimately it is up to the individual task to determine how to obtain
        a dictionary of target tensors to use for loss computation, but this
        implements the base logic assuming everything is neatly in the "targets"
        key of a batch.

        Parameters
        ----------
        batch : Dict[str, Union[torch.Tensor, dgl.DGLGraph, Dict[str, torch.Tensor]]]
            Batch of samples from the dataset.

        Returns
        -------
        Dict[str, torch.Tensor]
            A flat dictionary containing target tensors.
        """
        target_dict = {}
        assert len(self.task_keys) != 0, f"No target keys were set!"
        for key in self.task_keys:
            target_dict[key] = batch["targets"][key]
        return target_dict

    def _filter_task_keys(self, keys: List[str], batch: Dict[str, Union[torch.Tensor, dgl.DGLGraph, Dict[str, torch.Tensor]]]) -> List[str]:
        """
        Implement a mechanism for filtering out keys for targets.

        The base class simply returns the keys without modification.

        Parameters
        ----------
        keys : List[str]
            List of task keys
        batch : Dict[str, Union[torch.Tensor, dgl.DGLGraph, Dict[str, torch.Tensor]]]
            Batch of training samples to inspect.

        Returns
        -------
        List[str]
            List of filtered task keys
        """
        return keys

    def _compute_losses(
        self,
        batch: Dict[str, Union[torch.Tensor, dgl.DGLGraph, Dict[str, torch.Tensor]]],
    ) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Compute pred versus target for every target, then sum.

        Parameters
        ----------
        batch : Dict[str, Union[torch.Tensor, dgl.DGLGraph, Dict[str, torch.Tensor]]]
            Batch of samples to evaluate on.

        Returns
        -------
        Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]
            Dictionary containing the joint loss, and a subdictionary
            containing each individual target loss.
        """
        targets = self._get_targets(batch)
        predictions = self(batch)
        losses = {}
        for key in self.task_keys:
            losses[key] = self.loss_func(predictions[key], targets[key])
        total_loss: torch.Tensor = sum(losses.values())
        return {"loss": total_loss, "log": losses}

    def configure_optimizers(self) -> torch.optim.AdamW:
        opt = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        return opt

    def training_step(
        self,
        batch: Dict[str, Union[torch.Tensor, dgl.DGLGraph, Dict[str, torch.Tensor]]],
        batch_idx: int,
    ):
        loss_dict = self._compute_losses(batch)
        metrics = {}
        # prepending training flag for
        for key, value in loss_dict["log"].items():
            metrics[f"train_{key}"] = value
        if "graph" in batch.keys():
            batch_size = batch["graph"].batch_size
        else:
            batch_size = None
        self.log_dict(
            metrics, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size
        )
        return loss_dict

    def validation_step(
        self,
        batch: Dict[str, Union[torch.Tensor, dgl.DGLGraph, Dict[str, torch.Tensor]]],
        batch_idx: int,
    ):
        loss_dict = self._compute_losses(batch)
        metrics = {}
        # prepending training flag for
        for key, value in loss_dict["log"].items():
            metrics[f"val_{key}"] = value
        if "graph" in batch.keys():
            batch_size = batch["graph"].batch_size
        else:
            batch_size = None
        self.log_dict(metrics, on_epoch=True, batch_size=batch_size)
        return loss_dict

    def test_step(
        self,
        batch: Dict[str, Union[torch.Tensor, dgl.DGLGraph, Dict[str, torch.Tensor]]],
        batch_idx: int,
    ):
        loss_dict = self._compute_losses(batch)
        metrics = {}
        # prepending training flag for
        for key, value in loss_dict["log"].items():
            metrics[f"test_{key}"] = value
        if "graph" in batch.keys():
            batch_size = batch["graph"].batch_size
        else:
            batch_size = None
        self.log_dict(metrics, on_epoch=True, batch_size=batch_size)
        return loss_dict

    def _make_normalizers(self) -> Dict[str, Normalizer]:
        """
        Instantiate a set of normalizers for targets associated with this task.

        Assumes that task keys has been set correctly, and the default behavior
        will use normalizers with a mean and standard deviation of zero and one.

        Returns
        -------
        Dict[str, Normalizer]
            Normalizers for each target
        """
        if self.normalize_kwargs is not None:
            norm_kwargs = self.normalize_kwargs
        else:
            norm_kwargs = {}
        normalizers = {}
        for key in self.task_keys:
            mean = norm_kwargs.get(f"{key}_mean", 0.)
            std = norm_kwargs.get(f"{key}_std", 1.)
            normalizers[key] = Normalizer(mean=mean, std=std, device=self.device)
        return normalizers


class ScalarRegressionTask(BaseTaskModule):

    __task__ = "regression"

    """
    NOTE: You can have multiple targets, but each target is scalar.
    """

    def __init__(
        self,
        encoder: nn.Module,
        loss_func: Union[Type[nn.Module], nn.Module] = nn.MSELoss,
        task_keys: Optional[List[str]] = None,
        output_kwargs: Dict[str, Any] = {},
        **kwargs: Any,
    ) -> None:
        super().__init__(encoder, loss_func, task_keys, output_kwargs, **kwargs)
        self.save_hyperparameters(ignore=["encoder", "loss_func"])

    def _make_output_heads(self) -> nn.ModuleDict:
        modules = {}
        for key in self.task_keys:
            modules[key] = OutputHead(1, **self.output_kwargs).to(self.device)
        return nn.ModuleDict(modules)

    def _filter_task_keys(self, keys: List[str], batch: Dict[str, Union[torch.Tensor, dgl.DGLGraph, Dict[str, torch.Tensor]]]) -> List[str]:
        """
        Filters out task keys for scalar regression.

        This routine will filter out keys with targets that are multidimensional, since
        this is the _scalar_ regression task class.

        Parameters
        ----------
        keys : List[str]
            List of task keys
        batch : Dict[str, Union[torch.Tensor, dgl.DGLGraph, Dict[str, torch.Tensor]]]
            Batch of training samples to inspect.

        Returns
        -------
        List[str]
            List of filtered task keys
        """
        keys = super()._filter_task_keys(keys, batch)
        def checker(key) -> bool:
            # this ignores all non-tensor objects, and checks to make
            # sure the last target dimension is scalar
            target = batch["targets"][key]
            if isinstance(target, torch.Tensor):
                return target.size(-1) <= 1
            return False
        # this filters out targets that are multidimensional
        keys = list(filter(checker, keys))
        return keys

    def on_train_batch_start(
        self, batch: Any, batch_idx: int, unused: int = 0
    ) -> Optional[int]:
        """
        PyTorch Lightning hook to check OutputHeads are created.

        This will take data from the batch to determine which key to retrieve
        data from and how many heads to create.

        Parameters
        ----------
        batch : Dict[str, Union[torch.Tensor, dgl.DGLGraph, Dict[str, torch.Tensor]]]
            Batch of data from data loader.
        batch_idx : int
            Batch index.
        unused
            PyTorch Lightning hangover

        Returns
        -------
        Optional[int]
            Just returns the parent result.
        """
        status = super().on_train_batch_start(batch, batch_idx, unused)
        # if there are no task keys set, task has not been initialized yet
        if len(self.task_keys) == 0:
            keys = batch["target_types"]["regression"]
            self.task_keys = self._filter_task_keys(keys, batch)
            self.output_heads = self._make_output_heads()
            # now add the parameters to our task's optimizer
            opt = self.optimizers()
            opt.add_param_group({"params": self.output_heads.parameters()})
            # create normalizers for each target
            self.normalizers = self._make_normalizers()
        return status

    def on_validation_batch_start(
        self, batch: any, batch_idx: int, dataloader_idx: int
    ):
        self.on_train_batch_start(batch, batch_idx)


class BinaryClassificationTask(BaseTaskModule):

    __task__ = "classification"

    """
    Same as the regression case; you can have multiple targets,
    but each target has to be a binary classification task.

    Output heads will produce logits by default alongside BCEWithLogitsLoss
    for computation; if otherwise, requires user intervention.
    """

    def __init__(
        self,
        encoder: nn.Module,
        loss_func: Union[Type[nn.Module], nn.Module] = nn.BCEWithLogitsLoss,
        task_keys: Optional[List[str]] = None,
        output_kwargs: Dict[str, Any] = {},
        **kwargs,
    ) -> None:
        super().__init__(encoder, loss_func, task_keys, output_kwargs, **kwargs)
        self.save_hyperparameters(ignore=["encoder", "loss_func"])

    def _make_output_heads(self) -> nn.ModuleDict:
        modules = {}
        for key in self.task_keys:
            modules[key] = OutputHead(1, **self.output_kwargs).to(self.device)
        return nn.ModuleDict(modules)

    def on_train_batch_start(
        self, batch: Any, batch_idx: int, unused: int = 0
    ) -> Optional[int]:
        """
        PyTorch Lightning hook to check OutputHeads are created.

        This will take data from the batch to determine which key to retrieve
        data from and how many heads to create.

        Parameters
        ----------
        batch : Dict[str, Union[torch.Tensor, dgl.DGLGraph, Dict[str, torch.Tensor]]]
            Batch of data from data loader.
        batch_idx : int
            Batch index.
        unused
            PyTorch Lightning hangover

        Returns
        -------
        Optional[int]
            Just returns the parent result.
        """
        status = super().on_train_batch_start(batch, batch_idx, unused)
        # if there are no task keys set, task has not been initialized yet
        if len(self.task_keys) == 0:
            keys = batch["target_types"]["classification"]
            self.task_keys = keys
            self.output_heads = self._make_output_heads()
            # now add the parameters to our task's optimizer
            opt = self.optimizers()
            opt.add_param_group({"params": self.output_heads.parameters()})
        return status

    def on_validation_batch_start(
        self, batch: any, batch_idx: int, dataloader_idx: int
    ):
        self.on_train_batch_start(batch, batch_idx)


class MultiTaskLitModule(pl.LightningModule):
    def __init__(
        self,
        *tasks: Tuple[str, BaseTaskModule],
        task_scaling: Optional[Iterable[float]] = None,
        **encoder_opt_kwargs,
    ) -> None:
        """
        High level module for orchestrating multiple tasks.

        Keep in mind that multiple tasks is distinct from multiple datasets:
        this class can be used for multiple tasks even with a single dataset
        for example regression and classification in Materials Project.

        Parameters
        ----------
        *tasks : Tuple[str, BaseTaskModule]
            A variable number of 2-tuples, each comprising the
            dataset name and the task associated. Example would
            be ('MaterialsProjectDataset', RegressionTask).
        """
        super().__init__()
        assert len(tasks) > 0, f"No tasks provided."
        # hold a set of dataset mappings
        task_map = nn.ModuleDict()
        self.encoder = tasks[0][1].encoder
        dset_names = set()
        for index, entry in enumerate(tasks):
            # unpack tuple
            (dset_name, task) = entry
            if dset_name not in task_map:
                task_map[dset_name] = nn.ModuleDict()
            # set the task's encoder to be the same model instance except
            # the first to avoid recursion
            if index != 0:
                task.encoder = self.encoder
            # nest the task based on its category
            task_map[dset_name][task.__task__] = task
            # add dataset names to determine forward logic
            dset_names.add(dset_name)
        self.task_map = task_map
        self.dataset_names = dset_names
        self.task_scaling = task_scaling
        self.encoder_opt_kwargs = encoder_opt_kwargs
        self.automatic_optimization = False

    def configure_optimizers(self) -> List[Optimizer]:
        optimizers = []
        optimizer_names = []
        # iterate over tasks
        index = 0
        for data_key, tasks in self.task_map.items():
            for task_type, subtask in tasks.items():
                optimizer = subtask.configure_optimizers()
                optimizers.append(optimizer)
                optimizer_names.append((data_key, task_type))
                index += 1
        assert (
            len(optimizers) > 1
        ), f"Only one optimizer was found for multi-task training."
        opt_kwargs = {"lr": 1e-4}
        opt_kwargs.update(self.encoder_opt_kwargs)
        optimizers.append(AdamW(self.encoder.parameters(), **opt_kwargs))
        optimizer_names.append(("Global", "Encoder"))
        # this keeps a list of 2-tuples to index optimizers
        self.optimizer_names = optimizer_names
        return optimizers

    @property
    def dataset_names(self) -> List[str]:
        return self._dataset_names

    @dataset_names.setter
    def dataset_names(self, values: Union[set, List[str]]) -> None:
        if isinstance(values, set):
            values = list(values)
        self._dataset_names = values

    @property
    def task_scaling(self) -> List[float]:
        """
        Returns a list of scaling factors used task importance.

        These values are applied to the loss values prior to backprop.

        Returns
        -------
        List[float]
            List of scaling factors for each task
        """
        return self._task_scaling

    @task_scaling.setter
    def task_scaling(self, values: Union[Iterable[float], None]) -> None:
        if values is None:
            values = [1.0 for _ in range(self.num_tasks)]
        assert (
            len(values) == self.num_tasks
        ), f"Number of provided task scaling values not equal to number of tasks."
        self._task_scaling = values

    @property
    def num_tasks(self) -> int:
        """
        Return the total number of tasks.

        Returns
        -------
        int
            Number of tasks, aggregated over all datasets.
        """
        counter = 0
        # basically loop over datasets, and add up number of tasks
        # per dataset
        for tasks in self.task_map.values():
            counter += len(tasks)
        return counter

    @property
    def is_multidata(self) -> bool:
        # convenient property to determine how to unpack batches
        return len(self.dataset_names) > 1

    def forward(
        self,
        batch: Dict[
            str, Dict[str, Union[torch.Tensor, dgl.DGLGraph, Dict[str, torch.Tensor]]]
        ],
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Forward method for `MultiTaskLitModule`.

        Uses the number of unique dataset names (specified when creating)
        a `MultiTaskLitModule`) to determine what kind of batch structure
        is used. This might not be fully transparent, and might be something
        that needs to be refactored later.

        Parameters
        ----------
        batch
            [TODO:description]

        Returns
        -------
        Dict[str, Dict[str, torch.Tensor]]
            [TODO:description]
        """
        # iterate over datasets in the batch
        results = {}
        # for single dataset usage, we assume the nested structure isn't used
        if self.is_multidata:
            for key, data in batch.items():
                subtasks = self.task_map[key]
                if key not in results:
                    results[key] = {}
                # finally call the task with the data
                # TODO: refactor this to share an embedding, so the encoder
                # is only called once
                for task_type, subtask in subtasks.items():
                    results[key][task_type] = subtask(data)
        else:
            # in the single dataset case, we can skip the outer loop
            # and just pass the batch into the subtask
            tasks = list(self.task_map.values()).pop(0)
            for task_type, subtask in tasks.items():
                results[task_type] = subtask(batch)
        return results

    def on_train_batch_start(
        self, batch: Any, batch_idx: int, unused: int = 0
    ) -> Optional[int]:
        # this follows what's implemented in forward to ensure the
        # output heads and optimizers are set properly
        if self.is_multidata:
            for dataset in batch.keys():
                subtasks = self.task_map[dataset]
                for task_type in subtasks.keys():
                    self._initialize_subtask_output(dataset, task_type, batch)
        else:
            # skip grabbing dataset key from the batch
            tasks = list(self.task_map.values()).pop(0)
            dataset = list(self.task_map.keys()).pop(0)
            for task_type in tasks.keys():
                self._initialize_subtask_output(dataset, task_type, batch)
        return None

    def _compute_losses(
        self,
        batch: Dict[str, Union[torch.Tensor, dgl.DGLGraph, Dict[str, torch.Tensor]]],
    ):
        # compute predictions for required models
        losses = {}
        if self.is_multidata:
            for key, data in batch.items():
                subtasks = self.task_map[key]
                if key not in losses:
                    losses[key] = {}
                for task_type, subtask in subtasks.items():
                    losses[key][task_type] = subtask._compute_losses(data)
        else:
            tasks = list(self.task_map.values()).pop(0)
            for task_type, subtask in tasks.items():
                losses[task_type] = subtask._compute_losses(batch)
        return losses

    def _initialize_subtask_output(
        self,
        dataset: str,
        task_type: str,
        batch: Dict[
            str, Dict[str, Union[torch.Tensor, dgl.DGLGraph, Dict[str, torch.Tensor]]]
        ],
    ):
        """
        For a given dataset and task type, this function will check and initialize corresponding
        output heads and add them to the corresponding optimizer.

        [TODO:description]

        Parameters
        ----------
        dataset
            [TODO:description]
        task_type
            [TODO:description]
        batch
            [TODO:description]
        """
        task_instance = self.task_map[dataset][task_type]
        if not hasattr(task_instance, "output_head"):
            # get the task keys from the batch, depends on usage
            if self.is_multidata:
                subset = batch[dataset]
            else:
                subset = batch
            task_keys = subset["target_types"][task_type]
            # set task keys, then call make output heads
            task_instance.task_keys = task_instance._filter_task_keys(task_keys, subset)
            task_instance.output_heads = task_instance._make_output_heads()
            # now look up which optimizer it belongs to and add the parameters
            ref = (dataset, task_type)
            opt_index = self.optimizer_names.index(ref)
            # this adds the output head weights to optimizer
            self.optimizers()[opt_index].add_param_group(
                {"params": task_instance.output_heads.parameters()}
            )

    def embed(self, *args, **kwargs) -> Any:
        return self.encoder(*args, **kwargs)

    def _calculate_batch_size(
        self,
        batch: Dict[
            str, Dict[str, Union[torch.Tensor, dgl.DGLGraph, Dict[str, torch.Tensor]]]
        ],
    ) -> Dict[str, Union[int, Dict[str, int]]]:
        """
        Compute the size of a given batch.

        For multidata runs, this will sum over each of the subsets, providing a breakdown of
        how many samples from each respective dataset as well.

        Parameters
        ----------
        batch : Dict[str, Dict[str, Union[torch.Tensor, dgl.DGLGraph, Dict[str, torch.Tensor]]]]
            Batch of samples.

        Returns
        -------
        Dict[str, Union[int, Dict[str, int]]]
            Dictionary holding the batch size. For multidata runs, an additional "breakdown"
            key comprises the number of samples from each dataset.
        """
        batch_info = {}
        batch_size = 0
        if self.is_multidata:
            break_down = {}
            for dataset, subset in batch.items():
                # extract out targets to figure batch size for this subset of data
                key = next(iter(subset["targets"]))
                sample = subset["targets"][key]
                if isinstance(sample, dgl.DGLGraph):
                    counts = sample.batch_size
                elif isinstance(sample, torch.Tensor):
                    # assume first dimension is the batch size
                    counts = sample.size(0)
                else:
                    # assume the object is like a list
                    counts = len(sample)
                # track how much data from each dataset
                break_down[dataset] = counts
                batch_size += counts
            batch_info["breakdown"] = break_down
        else:
            key = next(iter(batch["targets"]))
            sample = batch["targets"][key]
            if isinstance(sample, dgl.DGLGraph):
                batch_size = sample.batch_size
            elif isinstance(sample, torch.Tensor):
                # assume first dimension is the batch size
                batch_size = sample.size(0)
            else:
                # assume the object is like a list
                batch_size = len(sample)
        batch_info["batch_size"] = batch_size
        return batch_info

    def __repr__(self) -> str:
        build_str = "MultiTask Training module:\n"
        for dataset, tasks in self.task_map.items():
            for task_type in tasks.keys():
                build_str += f"{dataset}-{task_type}\n"
        return build_str

    def training_step(
        self,
        batch: Dict[
            str, Dict[str, Union[torch.Tensor, dgl.DGLGraph, Dict[str, torch.Tensor]]]
        ],
        batch_idx: int,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Manual training logic for multi tasks.

        We sequentially step through each loss returned, and perform
        backpropagation. The logic looks complicated, because we have
        to match each loss with its corresponding optimizer.

        Parameters
        ----------
        batch : Dict[str, Dict[str, Union[torch.Tensor, dgl.DGLGraph, Dict[str, torch.Tensor]]]]
            Batch of data from one or more datasets.
        batch_idx : int
            Index of current batch
        """
        # zero all gradients
        optimizers = self.optimizers()
        for opt in optimizers:
            self.on_before_zero_grad(opt)
            opt.zero_grad()
        losses = self._compute_losses(batch)
        loss_logging = {}
        # for multiple datasets, we step through each dataset
        if self.is_multidata:
            for dataset_name, task_loss in losses.items():
                for task_name, subtask_loss in task_loss.items():
                    # get the right optimizer by indexing our lookup list
                    ref = (dataset_name, task_name)
                    opt_index = self.optimizer_names.index(ref)
                    # backprop gradients
                    opt = optimizers[opt_index]
                    opt = optimizers[opt_index]
                    is_last_opt = opt_index == len(self.optimizer_names) - 2
                    # run hooks between backward
                    self.on_before_backward(subtask_loss["loss"])
                    # scale loss values in task
                    scaling = self.task_scaling[opt_index]
                    self.manual_backward(
                        subtask_loss["loss"] * scaling, retain_graph=not is_last_opt
                    )
                    self.on_after_backward()
                    loss_logging.update(subtask_loss["log"])
        # for single dataset, we can just unpack the dictionary directly
        else:
            dataset_name = self.dataset_names[0]
            for task_name, loss in losses.items():
                opt_index = self.optimizer_names.index((dataset_name, task_name))
                opt = optimizers[opt_index]
                is_last_opt = opt_index == len(self.optimizer_names) - 2
                # run hooks between backward
                self.on_before_backward(loss["loss"])
                # scale loss values in task
                scaling = self.task_scaling[opt_index]
                self.manual_backward(
                    loss["loss"] * scaling, retain_graph=not is_last_opt
                )
                self.on_after_backward()
                loss_logging.update(loss["log"])
        # run before step hooks
        for opt_idx, opt in enumerate(optimizers):
            self.on_before_optimizer_step(opt, opt_idx)
            opt.step()
        # add train prefix to metric logs
        prepend_affix(loss_logging, "train")
        batch_info = self._calculate_batch_size(batch)
        if "breakdown" in batch_info:
            for key, value in batch_info["breakdown"].items():
                self.log(f"{key}.num_samples", float(value), on_step=True, on_epoch=False, reduce_fx="min")
        self.log_dict(loss_logging, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_info["batch_size"])
        return losses
