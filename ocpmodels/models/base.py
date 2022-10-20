# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: MIT License

from typing import Dict, Type, Tuple, Optional, Union, ContextManager
from abc import abstractmethod
from contextlib import nullcontext, ExitStack
import logging

import pytorch_lightning as pl
import torch
from torch import Tensor, nn
import dgl

from ocpmodels.modules.normalizer import Normalizer

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
            # self._nan_logger.setLevel(logging.DEBUG)
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
                # on_step=True,
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
        energy = self.gnn(graph, *args, **kwargs)
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
        with dynamic_gradients_context(
            self.regress_forces, self.has_rnn
        ) as grad_context:
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

    def step(
        self,
        batch: Dict[str, Union[torch.Tensor, dgl.DGLGraph]],
        batch_idx: int,
        prefix: str = "validation",
    ) -> float:
        """
        Wraps the parent class's generic step method with a gradient context.

        This ensures that derivatives are computable (i.e. forces) even outside
        of training mode/steps.

        Parameters
        ----------
        batch : Dict[str, Union[torch.Tensor, dgl.DGLGraph]]
            Dictionary containing batched data
        batch_idx : int
            Index of this batch
        """
        with dynamic_gradients_context(
            self.regress_forces, self.has_rnn
        ) as grad_context:
            return super().step(batch, batch_idx, prefix)

    def on_validation_start(self) -> None:
        """
        Configures RNN subclasses to be in training mode for CUDA enabled
        systems.
        """
        self.apply(rnn_force_train_mode)

    def on_test_start(self) -> None:
        """
        Configures RNN subclasses to be in training mode for CUDA enabled
        systems.
        """
        self.apply(rnn_force_train_mode)

    def on_validation_batch_end(
        self,
        outputs,
        batch: Dict[str, Union[torch.Tensor, dgl.DGLGraph]],
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        # make sure gradients are zeroed out and they don't contaminate
        # the next batch
        self.zero_grad()

    def on_test_batch_end(
        self,
        outputs,
        batch: Dict[str, Union[torch.Tensor, dgl.DGLGraph]],
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        # make sure gradients are zeroed out and they don't contaminate
        # the next batch
        self.zero_grad()


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
        energy = self.model(features, positions, *args, **kwargs)
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
