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
from abc import abstractmethod, ABC
from contextlib import nullcontext, ExitStack
import logging
from warnings import warn
from dgl.utils import data

import pytorch_lightning as pl
import torch
from torch import Tensor, nn
import dgl
from torch.optim import AdamW, Optimizer

from ocpmodels.modules.normalizer import Normalizer
from ocpmodels.models.common import OutputHead
from ocpmodels.common.types import DataDict, BatchDict

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
    except AssertionError:
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
        metrics[f"{affix}.{key}"] = metrics[key]
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


class AbstractEnergyModel(AbstractTask):
    __task__ = "S2EF"

    """
    At a minimum, the point of this is to help register associated models
    with PyTorch Lightning ModelRegistry; the expectation is that you get
    the graph energy as well as the atom forces.
    """

    def __init__(self):
        super().__init__()
        self.save_hyperparameters()

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
    __needs_grads__ = []

    def __init__(
        self,
        encoder: Optional[nn.Module] = None,
        encoder_class: Optional[Type[nn.Module]] = None,
        encoder_kwargs: Optional[Dict[str, Any]] = None,
        loss_func: Optional[Union[Type[nn.Module], nn.Module]] = None,
        task_keys: Optional[List[str]] = None,
        output_kwargs: Dict[str, Any] = {},
        lr: float = 1e-4,
        weight_decay: float = 0.0,
        normalize_kwargs: Optional[Dict[str, float]] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        if encoder is not None:
            warn(
                f"Encoder object was passed directly into {self.__class__.__name__}; saved hyperparameters will be incomplete!"
            )
        if encoder_class is not None and encoder_kwargs:
            try:
                encoder = encoder_class(**encoder_kwargs)
            except:
                raise ValueError(
                    f"Unable to instantiate encoder {encoder_class} with kwargs: {encoder_kwargs}."
                )
        if encoder is not None:
            self.encoder = encoder
        else:
            raise ValueError(f"No valid encoder passed.")
        if isinstance(loss_func, Type):
            loss_func = loss_func()
        self.loss_func = loss_func
        default_heads = {"act_last": None, "hidden_dim": 128}
        default_heads.update(output_kwargs)
        self.output_kwargs = default_heads
        self.normalize_kwargs = normalize_kwargs
        self.task_keys = task_keys
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
        # if we're setting task keys we have enough to initialize
        # the output heads
        if not self.has_initialized:
            self.output_heads = self._make_output_heads()
            self.normalizers = self._make_normalizers()
        self.hparams["task_keys"] = self._task_keys

    @property
    def has_initialized(self) -> bool:
        if len(self.task_keys) == 0:
            return False
        output_heads = getattr(self, "output_heads", None)
        if output_heads is None:
            return False
        # basically if we've passed these two assertions, we should have
        # all the heads. We can't check against self.task_keys, because
        # some tasks like ForceRegressionTask doesn't actually use an output
        # head for the forces
        return True

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

    @property
    def uses_normalizers(self) -> bool:
        # property determines if we normalize targets or not
        norms = getattr(self, "normalizers", None)
        if norms is None or self.__task__ in ["classification", "symmetry"]:
            return False
        return True

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

    def forward(
        self,
        batch: Dict[str, Union[torch.Tensor, dgl.DGLGraph, Dict[str, torch.Tensor]]],
    ) -> Dict[str, torch.Tensor]:
        if "embeddings" in batch:
            embedding = batch.get("embeddings")
        else:
            embedding = self.encoder(batch)
        outputs = self.process_embedding(embedding)
        return outputs

    def process_embedding(self, embeddings: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Given a set of embeddings, output predictions for each head.

        Parameters
        ----------
        embeddings : torch.Tensor
            Batch of graph/point cloud embeddings

        Returns
        -------
        Dict[str, torch.Tensor]
            Predictions per output head
        """
        results = {}
        for key, head in self.output_heads.items():
            results[key] = head(embeddings)
        return results

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

    def _filter_task_keys(
        self,
        keys: List[str],
        batch: Dict[str, Union[torch.Tensor, dgl.DGLGraph, Dict[str, torch.Tensor]]],
    ) -> List[str]:
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

        embeddings : Optional[torch.Tensor]
            If provided, bypasses calling the encoder and obtains predictions
            from processing the embeddings. Mainly intended for use with multitask
            abstraction.

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
            target_val = targets[key]
            if self.uses_normalizers:
                target_val = self.normalizers[key].norm(target_val)
            losses[key] = self.loss_func(predictions[key], target_val)
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
        self.log_dict(metrics, on_step=True, prog_bar=True, batch_size=batch_size)
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
        self.log_dict(metrics, batch_size=batch_size)
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
        self.log_dict(metrics, batch_size=batch_size)
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
            mean = norm_kwargs.get(f"{key}_mean", 0.0)
            std = norm_kwargs.get(f"{key}_std", 1.0)
            normalizers[key] = Normalizer(mean=mean, std=std, device=self.device)
        return normalizers


class ScalarRegressionTask(BaseTaskModule):
    __task__ = "regression"

    """
    NOTE: You can have multiple targets, but each target is scalar.
    """

    def __init__(
        self,
        encoder: Optional[nn.Module] = None,
        encoder_class: Optional[Type[nn.Module]] = None,
        encoder_kwargs: Optional[Dict[str, Any]] = None,
        loss_func: Union[Type[nn.Module], nn.Module] = nn.MSELoss,
        task_keys: Optional[List[str]] = None,
        output_kwargs: Dict[str, Any] = {},
        **kwargs: Any,
    ) -> None:
        super().__init__(
            encoder,
            encoder_class,
            encoder_kwargs,
            loss_func,
            task_keys,
            output_kwargs,
            **kwargs,
        )
        self.save_hyperparameters(ignore=["encoder", "loss_func"])

    def _make_output_heads(self) -> nn.ModuleDict:
        modules = {}
        for key in self.task_keys:
            modules[key] = OutputHead(1, **self.output_kwargs).to(self.device)
        return nn.ModuleDict(modules)

    def _filter_task_keys(
        self,
        keys: List[str],
        batch: Dict[str, Union[torch.Tensor, dgl.DGLGraph, Dict[str, torch.Tensor]]],
    ) -> List[str]:
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

    def on_train_batch_start(self, batch: Any, batch_idx: int) -> Optional[int]:
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
        status = super().on_train_batch_start(batch, batch_idx)
        # if there are no task keys set, task has not been initialized yet
        if len(self.task_keys) == 0:
            keys = batch["target_types"]["regression"]
            self.task_keys = self._filter_task_keys(keys, batch)
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
        encoder: Optional[nn.Module] = None,
        encoder_class: Optional[Type[nn.Module]] = None,
        encoder_kwargs: Optional[Dict[str, Any]] = None,
        loss_func: Union[Type[nn.Module], nn.Module] = nn.BCEWithLogitsLoss,
        task_keys: Optional[List[str]] = None,
        output_kwargs: Dict[str, Any] = {},
        **kwargs,
    ) -> None:
        super().__init__(
            encoder,
            encoder_class,
            encoder_kwargs,
            loss_func,
            task_keys,
            output_kwargs,
            **kwargs,
        )
        self.save_hyperparameters(ignore=["encoder", "loss_func"])

    def _make_output_heads(self) -> nn.ModuleDict:
        modules = {}
        for key in self.task_keys:
            modules[key] = OutputHead(1, **self.output_kwargs).to(self.device)
        return nn.ModuleDict(modules)

    def on_train_batch_start(self, batch: Any, batch_idx: int) -> Optional[int]:
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
            # now add the parameters to our task's optimizer
            opt = self.optimizers()
            opt.add_param_group({"params": self.output_heads.parameters()})
        return status

    def on_validation_batch_start(
        self, batch: Any, batch_idx: int, dataloader_idx: int
    ):
        self.on_train_batch_start(batch, batch_idx)


class ForceRegressionTask(BaseTaskModule):
    __task__ = "regression"
    __needs_grads__ = ["pos"]

    def __init__(
        self,
        encoder: Optional[nn.Module] = None,
        encoder_class: Optional[Type[nn.Module]] = None,
        encoder_kwargs: Optional[Dict[str, Any]] = None,
        loss_func: Union[Type[nn.Module], nn.Module] = nn.L1Loss,
        task_keys: Optional[List[str]] = None,
        output_kwargs: Dict[str, Any] = {},
        **kwargs,
    ) -> None:
        super().__init__(
            encoder,
            encoder_class,
            encoder_kwargs,
            loss_func,
            task_keys,
            output_kwargs,
            **kwargs,
        )
        self.save_hyperparameters(ignore=["encoder", "loss_func"])
        # have to enable double backprop
        self.automatic_optimization = False

    def _make_output_heads(self) -> nn.ModuleDict:
        # this task only utilizes one output head
        modules = {"energy": OutputHead(1, **self.output_kwargs).to(self.device)}
        return nn.ModuleDict(modules)

    def forward(
        self,
        batch: Dict[str, Union[torch.Tensor, dgl.DGLGraph, Dict[str, torch.Tensor]]],
    ) -> Dict[str, torch.Tensor]:
        # for ease of use, this task will always compute forces
        with dynamic_gradients_context(True, self.has_rnn):
            # first ensure that positions tensor is backprop ready
            if "graph" in batch:
                pos: torch.Tensor = batch["graph"].ndata.get("pos")
            else:
                # assume point cloud otherwise
                pos: torch.Tensor = batch.get("pos")
            if pos is None:
                raise ValueError(
                    f"No atomic positions were found in batch - neither as standalone tensor nor graph."
                )
            pos.requires_grad_(True)
            if "embeddings" in batch:
                embeddings = batch.get("embeddings")
            else:
                embeddings = self.encoder(batch)
            outputs = self.process_embedding(embeddings, pos)
        return outputs

    def process_embedding(
        self, embeddings: torch.Tensor, pos: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        outputs = {}
        energy = self.output_heads["energy"](embeddings)
        # now use autograd for force calculation
        force = (
            -1
            * torch.autograd.grad(
                energy,
                pos,
                grad_outputs=torch.ones_like(energy),
                create_graph=True,
            )[0]
        )
        outputs["force"] = force
        outputs["energy"] = energy
        return outputs

    def _get_targets(
        self,
        batch: Dict[str, Union[torch.Tensor, dgl.DGLGraph, Dict[str, torch.Tensor]]],
    ) -> Dict[str, torch.Tensor]:
        """
        Extract out the energy and force targets from a batch.

        The intended behavior is similar to other tasks, however explicit because
        we actually expect "energy" and "force" keys as opposed to inferring them from a batch.

        Parameters
        ----------
        batch : Dict[str, Union[torch.Tensor, dgl.DGLGraph, Dict[str, torch.Tensor]]]
            Batch of samples to evaluate

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary containing targets to evaluate against

        Raises
        ------
        KeyError
            If either "energy" or "force" keys aren't found in the "targets"
            dictionary within a batch, we abort the program.
        """
        target_dict = {}
        for key in ["energy", "force"]:
            try:
                target_dict[key] = batch["targets"][key]
            except KeyError as e:
                raise KeyError(
                    f"{key} was not found in targets key in batch, which is needed for force regression task."
                ) from e
        return target_dict

    def on_train_batch_start(self, batch: Any, batch_idx: int) -> Optional[int]:
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
        status = super().on_train_batch_start(batch, batch_idx)
        # if there are no task keys set, task has not been initialized yet
        if len(self.task_keys) == 0:
            # first round is used to initialize the output head
            self.task_keys = ["energy"]
            self.output_heads = self._make_output_heads()
            # overwrite it so that the loss is computed but we don't make another head
            # for force outputs
            self._task_keys = ["energy", "force"]
            # now add the parameters to our task's optimizer
            opt = self.optimizers()
            opt.add_param_group({"params": self.output_heads.parameters()})
            # create normalizers for each target
            self.normalizers = self._make_normalizers()
        return status

    def training_step(
        self,
        batch: Dict[str, Union[torch.Tensor, dgl.DGLGraph, Dict[str, torch.Tensor]]],
        batch_idx: int,
    ):
        """
        Implements the training logic for force regression.

        This task uses manual optimization to facilitate double backprop, but by
        in large functions in the same way as other tasks.

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
        opt = self.optimizers()
        self.on_before_zero_grad(opt)
        opt.zero_grad()
        # compute losses
        loss_dict = self._compute_losses(batch)
        loss = loss_dict["loss"]
        # sandwich lightning callbacks
        self.manual_backward(loss, retain_graph=True)
        self.manual_backward(loss)
        self.on_before_optimizer_step(opt, 0)
        opt.step()
        metrics = {}
        # prepending training flag
        for key, value in loss_dict["log"].items():
            metrics[f"train_{key}"] = value
        if "graph" in batch.keys():
            batch_size = batch["graph"].batch_size
        else:
            batch_size = None
        self.log_dict(metrics, on_step=True, prog_bar=True, batch_size=batch_size)
        return loss_dict


class CrystalSymmetryClassificationTask(BaseTaskModule):
    __task__ = "symmetry"

    def __init__(
        self,
        encoder: Optional[nn.Module] = None,
        encoder_class: Optional[Type[nn.Module]] = None,
        encoder_kwargs: Optional[Dict[str, Any]] = None,
        loss_func: Union[Type[nn.Module], nn.Module] = nn.CrossEntropyLoss,
        output_kwargs: Dict[str, Any] = {},
        lr: float = 0.0001,
        weight_decay: float = 0,
        normalize_kwargs: Optional[Dict[str, float]] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            encoder,
            encoder_class,
            encoder_kwargs,
            loss_func,
            [
                "spacegroup",
            ],
            output_kwargs,
            lr,
            weight_decay,
            normalize_kwargs,
            **kwargs,
        )

    def _make_output_heads(self) -> nn.ModuleDict:
        # this task only utilizes one output head; 230 possible space groups
        modules = {"spacegroup": OutputHead(230, **self.output_kwargs).to(self.device)}
        return nn.ModuleDict(modules)

    def on_train_batch_start(self, batch: Any, batch_idx: int) -> Optional[int]:
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
        status = super().on_train_batch_start(batch, batch_idx)
        # if there are no task keys set, task has not been initialized yet
        if len(self.task_keys) == 0:
            self.task_keys = [
                "spacegroup",
            ]
            # now add the parameters to our task's optimizer
            opt = self.optimizers()
            opt.add_param_group({"params": self.output_heads.parameters()})
        return status

    def on_validation_batch_start(
        self, batch: Any, batch_idx: int, dataloader_idx: int
    ):
        self.on_train_batch_start(batch, batch_idx)

    def _get_targets(
        self,
        batch: Dict[str, Union[torch.Tensor, dgl.DGLGraph, Dict[str, torch.Tensor]]],
    ) -> Dict[str, torch.Tensor]:
        target_dict = {}
        subdict = batch.get("symmetry", None)
        if subdict is None:
            raise ValueError(
                f"'symmetry' key is missing from batch, which is needed for space group classification."
            )
        labels: torch.Tensor = subdict.get("number", None)
        if labels is None:
            raise ValueError(
                "Point group numbers missing from symmetry key, which is needed for symmetry classification."
            )
        # subtract one for zero-indexing
        labels = labels.long() - 1
        # cast to long type, and make sure it is 1D for cross entropy loss
        if labels.ndim > 1:
            labels = labels.flatten()
        target_dict["spacegroup"] = labels
        return target_dict


class MultiTaskLitModule(pl.LightningModule):
    def __init__(
        self,
        *tasks: Tuple[str, BaseTaskModule],
        task_scaling: Optional[Iterable[float]] = None,
        task_keys: Optional[Dict[str, List[str]]] = None,
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
        subtask_hparams = {}
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
            # save hyperparameters from subtasks
            subtask_hparams[f"{dset_name}_{task.__class__.__name__}"] = task.hparams
        self.save_hyperparameters(
            {
                "subtask_hparams": subtask_hparams,
                "task_scaling": task_scaling,
                "encoder_opt_kwargs": encoder_opt_kwargs,
            }
        )
        self.task_map = task_map
        self.dataset_names = dset_names
        self.task_scaling = task_scaling
        self.encoder_opt_kwargs = encoder_opt_kwargs
        if task_keys is not None:
            for pair in self.dataset_task_pairs:
                # unpack 2-tuple
                dataset_name, task_type = pair
                relevant_keys = task_keys[dataset_name][task_type]
                self._initialize_subtask_output(
                    dataset_name, task_type, task_keys=relevant_keys
                )
        self.configure_optimizers()
        self.automatic_optimization = False

    @property
    def task_list(self) -> List[BaseTaskModule]:
        # return a flat list of tasks to iterate over
        modules = []
        for task_group in self.task_map.values():
            for subtask in task_group.values():
                modules.append(subtask)
        return modules

    @property
    def dataset_task_pairs(self) -> List[Tuple[str, str]]:
        # Return a list of 2-tuples corresponding to (dataset name, task type)
        pairs = []
        for dataset in self.dataset_names:
            task_types = self.task_map[dataset].keys()
            for task_type in task_types:
                pairs.append((dataset, task_type))
        return pairs

    def configure_optimizers(self) -> List[Optimizer]:
        """
        Configure subtask optimizers, as well as the joint encoder optimizer.

        The main logic of this function is to aggregate all of the subtask
        optimizers together, if they haven't been added yet. This is done
        by assuming dataset name/task type combinations are unique, and we
        rely on the subtask's own `configure_optimizers` function.

        The latter half of the function adds the encoder optimizer.

        Returns
        -------
        List[Optimizer]
            List of optimizers that are subsequently passed into Lightning's
            internal mechanisms
        """
        optimizers = []
        # this keeps a list of 2-tuples to index optimizers
        self.optimizer_names = []
        # iterate over tasks
        index = 0
        for data_key, tasks in self.task_map.items():
            for task_type, subtask in tasks.items():
                combo = (data_key, task_type)
                if combo not in self.optimizer_names:
                    output_head = getattr(subtask, "output_heads", None)
                    assert (
                        output_head is not None
                    ), f"{subtask} does not contain output heads; ensure `task_keys` are set: {subtask.task_keys}"
                    optimizer = subtask.configure_optimizers()
                    # remove all the optimizer parameters, and re-add only the output heads
                    optimizer.param_groups.clear()
                    optimizer.add_param_group({"params": output_head.parameters()})
                    # add optimizer to the pile
                    optimizers.append(optimizer)
                    self.optimizer_names.append((data_key, task_type))
                    index += 1
        assert (
            len(self.optimizer_names) > 1
        ), f"Only one optimizer was found for multi-task training."
        if ("Global", "Encoder") not in self.optimizer_names:
            opt_kwargs = {"lr": 1e-4}
            opt_kwargs.update(self.encoder_opt_kwargs)
            optimizers.append(AdamW(self.encoder.parameters(), **opt_kwargs))
            self.optimizer_names.append(("Global", "Encoder"))
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

    @property
    def has_initialized(self) -> bool:
        """
        Property to track if subtasks have been initialized.

        Right now this is manually set, but would like to refactor this later to
        check if subtask output heads are all set.

        Returns
        -------
        bool
            True if first batch has been run already, otherwise False
        """
        return all([task.has_initialized for task in self.task_list])

    @property
    def input_grad_keys(self) -> Dict[str, List[str]]:
        """
        Property to returns a list of keys for inputs that need gradient tracking.

        Returns
        -------
        Union[List[str], None]
            If there are tasks in this multitask that need input variables to have
            gradients tracked, this property will return a list of them. Otherwise,
            this returns None.
        """
        keys = {}
        if self.is_multidata:
            for dset_name, task_group in self.task_map.items():
                if dset_name not in keys:
                    keys[dset_name] = set()
                dset_keyset = keys.get(dset_name)
                for subtask in task_group.values():
                    dset_keyset.update(subtask.__needs_grads__)
        else:
            tasks = list(self.task_map.values()).pop(0)
            keys[self.dataset_names[0]] = set()
            for task in tasks:
                keys[self.dataset_names[0]].update(task.__needs_grads__)
        keys = {dset_name: sorted(keys) for dset_name, keys in keys.items()}
        return keys

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

    @property
    def needs_dynamic_grads(self) -> bool:
        """
        Boolean property reflecting whether this multitask in general needs
        gradient computation to override inference modes.

        Returns
        -------
        bool
            True if any datasets need input grads, otherwise False
        """
        return sum([len(keys) for keys in self.input_grad_keys.values()]) > 0

    def _toggle_input_grads(
        self,
        batch: Dict[
            str, Dict[str, Union[torch.Tensor, dgl.DGLGraph, Dict[str, torch.Tensor]]]
        ],
    ) -> None:
        """
        Inplace method that will automatically enable gradient tracking for tensors
        needed by tasks/datasets.

        This function will loop over a batch of data (in the multidata case) and
        grabs the list of tensor keys as required by a given subtask. The list
        of tensor keys are then used to grab the input data from the batch and/or
        graph, and if it's found will then try and set requires_grad_(True).

        Parameters
        ----------
        batch : Dict[str, Dict[str, Union[torch.Tensor, dgl.DGLGraph, Dict[str, torch.Tensor]]]]
            Batch of data
        """
        need_grad_keys = getattr(self, "input_grad_keys", None)
        if need_grad_keys is not None:
            if self.is_multidata:
                # if this is a multidataset task, loop over each dataset
                # and enable gradients for the inputs that need them
                for dset_name, data in batch.items():
                    input_keys = need_grad_keys.get(dset_name)
                    for key in input_keys:
                        # set require grad for both point cloud and graph tensors
                        try:
                            if "graph" in data:
                                data["graph"].ndata[key].requires_grad_(True)
                            if key in data:
                                data[key].requires_grad_(True)
                        except KeyError:
                            pass
            else:
                # in the single dataset case, we just need to loop over a single
                # set of tasks
                input_keys = list(self.input_grad_keys.values()).pop(0)
                for key in input_keys:
                    try:
                        if "graph" in batch:
                            batch["graph"].ndata[key].requires_grad_(True)
                        if key in batch:
                            batch[key].requires_grad_(True)
                    except KeyError:
                        pass

    def forward(
        self,
        batch: Dict[
            str, Dict[str, Union[torch.Tensor, dgl.DGLGraph, Dict[str, torch.Tensor]]]
        ],
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Forward method for `MultiTaskLitModule`.

        This is devised slightly specially to comprise a variety of scenarios, including
        wrapping the entire compute in gradient contexts (for force prediction tasks),
        ensuring inputs that need gradients are enabled, as well as running the
        encoder at the beginning and passing the embeddings onto downstream tasks.

        Parameters
        ----------
        batch : Dict[str, Dict[str, Union[torch.Tensor, dgl.DGLGraph, Dict[str, torch.Tensor]]]]
            Batches of samples per dataset

        Returns
        -------
        Dict[str, Dict[str, torch.Tensor]]
            Dictionary of predictions, per dataset per subtask
        """
        # iterate over datasets in the batch
        results = {}
        _grads = getattr(
            self, "needs_dynamic_grads", False
        )  # default to not needing grads
        with dynamic_gradients_context(_grads, self.has_rnn):
            # this function switches of `requires_grad_` for input tensors that need them
            self._toggle_input_grads(batch)
            # compute embeddings for each dataset
            if self.is_multidata:
                for key, data in batch.items():
                    data["embeddings"] = self.encoder(data)
            else:
                batch["embeddings"] = self.encoder(batch)
            # for single dataset usage, we assume the nested structure isn't used
            if self.is_multidata:
                for key, data in batch.items():
                    subtasks = self.task_map[key]
                    if key not in results:
                        results[key] = {}
                    # finally call the task with the data
                    for task_type, subtask in subtasks.items():
                        results[key][task_type] = subtask(data)
            else:
                # in the single dataset case, we can skip the outer loop
                # and just pass the batch into the subtask
                tasks = list(self.task_map.values()).pop(0)
                for task_type, subtask in tasks.items():
                    results[task_type] = subtask(batch)
            return results

    def on_train_batch_start(self, batch: Any, batch_idx: int) -> None:
        """
        This callback is used to dynamically initialize output heads.

        In the event where `task_keys` are not explicitly provided by the user
        into the creation of each task, we the incoming batch for tasks
        that have not been initialized and create the output heads.

        Parameters
        ----------
        batch : Dict[str, Union[torch.Tensor, dgl.DGLGraph, Dict[str, torch.Tensor]]]
            Batch of samples to compute
        batch_idx : int
            Batch index
        unused : int
            Legacy PyTorch Lightning arg
        """
        # this follows what's implemented in forward to ensure the
        # output heads and optimizers are set properly
        if not self.has_initialized:
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
        """
        Function for computing the losses over a batch.

        This relies on the `_compute_losses` function of each subtask. Between the single
        dataset and multidataset settings, the difference is just how the tasks are retrieved;
        the former skips going through the dataset/task hierarchy.

        Parameters
        ----------
        batch : Dict[str, Union[torch.Tensor, dgl.DGLGraph, Dict[str, torch.Tensor]]]
            Batch of samples to calculate losses over
        """
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
        batch: Optional[
            Dict[
                str,
                Dict[str, Union[torch.Tensor, dgl.DGLGraph, Dict[str, torch.Tensor]]],
            ]
        ] = None,
        task_keys: Optional[List[str]] = None,
    ):
        """
        For a given dataset and task type, this function will check and initialize corresponding
        output heads and add them to the corresponding optimizer.

        The behavior of this function changes depending on whether or not the output heads were
        initialized earlier (i.e. before `on_train_batch_start`), based on whether it sees an
        incoming batch, or explicitly passed `task_keys`. In the former, we will add the output
        head parameters to the appropriate optimizer as well.

        Parameters
        ----------
        dataset : str
            Name of the dataset
        task_type : str
            String classification of the task type, e.g. "regression"
        batch : Optional[Dict[str, Dict[str, Union[torch.Tensor, dgl.DGLGraph, Dict[str, torch.Tensor]]]]]
            For "dynamically" instantiating multitasks, this function relies on an incoming batch
            to determine what output heads to instantiate.
        """
        task_instance: BaseTaskModule = self.task_map[dataset][task_type]
        if batch is None and task_keys is None:
            raise ValueError(
                f"Unable to initialize output heads for {dataset}-{task_type}; neither batch nor task keys provided."
            )
        if not task_instance.has_initialized:
            # get the task keys from the batch, depends on usage
            if batch is not None:
                if self.is_multidata:
                    subset = batch[dataset]
                else:
                    subset = batch
            if task_keys is None:
                task_keys = subset["target_types"][task_type]
                # if keys aren't explicitly provided, apply filter
                task_keys = task_instance._filter_task_keys(task_keys, subset)
                # set task keys, then call make output heads
            task_instance.task_keys = task_keys
            if task_type == "regression":
                task_instance.normalizers = task_instance._make_normalizers()
            if batch is not None:
                # if batch was provided then this is done after configure_optimizers
                # so we need to add their parameters to the right optimizer
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
                if "graph" in subset:
                    counts = subset["graph"].batch_size
                elif len(subset["targets"]) > 0:
                    key = next(iter(batch["targets"]))
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
            if "graph" in batch:
                batch_size = batch["graph"].batch_size
            elif len(subset["targets"]) > 0:
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
            opt.zero_grad(set_to_none=True)
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
                    is_last_opt = opt_index == len(self.optimizer_names) - 2
                    # run hooks between backward
                    self.on_before_backward(subtask_loss["loss"])
                    # scale loss values in task
                    scaling = self.task_scaling[opt_index]
                    self.manual_backward(
                        subtask_loss["loss"] * scaling, retain_graph=not is_last_opt
                    )
                    self.on_after_backward()
                    prepend_affix(subtask_loss["log"], dataset_name)
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
        # compoute the joint loss for logging purposes
        loss_logging["total_loss"] = sum(list(loss_logging.values()))
        # add train prefix to metric logs
        prepend_affix(loss_logging, "train")
        batch_info = self._calculate_batch_size(batch)
        if "breakdown" in batch_info:
            for key, value in batch_info["breakdown"].items():
                self.log(
                    f"{key}.num_samples",
                    float(value),
                    on_step=True,
                    on_epoch=False,
                    reduce_fx="min",
                )
        self.log_dict(
            loss_logging,
            on_step=True,
            prog_bar=True,
            batch_size=batch_info["batch_size"],
        )
        return losses

    def validation_step(
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
        losses = self._compute_losses(batch)
        loss_logging = {}
        # for multiple datasets, we step through each dataset
        if self.is_multidata:
            for dataset_name, task_loss in losses.items():
                for task_name, subtask_loss in task_loss.items():
                    prepend_affix(subtask_loss["log"], dataset_name)
                    loss_logging.update(subtask_loss["log"])
        # for single dataset, we can just unpack the dictionary directly
        else:
            dataset_name = self.dataset_names[0]
            for task_name, loss in losses.items():
                loss_logging.update(loss["log"])
        # compoute the joint loss for logging purposes
        loss_logging["total_loss"] = sum(list(loss_logging.values()))
        # add train prefix to metric logs
        prepend_affix(loss_logging, "val")
        batch_info = self._calculate_batch_size(batch)
        if "breakdown" in batch_info:
            for key, value in batch_info["breakdown"].items():
                self.log(
                    f"{key}.num_samples",
                    float(value),
                    on_epoch=True,
                    reduce_fx="min",
                )
        self.log_dict(
            loss_logging,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_info["batch_size"],
        )
        return losses

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path,
        map_location=None,
        hparams_file=None,
        strict: bool = True,
        **kwargs: Any,
    ):
        raise NotImplementedError(
            f"MultiTask should be reloaded using the `ocpmodels.models.multitask_from_checkpoint` function instead."
        )


class OpenCatalystInference(ABC, pl.LightningModule):
    """
    Implement a set of bare bones LightningModules that are solely used
    for OpenCatalyst leaderboard submissions.
    """

    def __init__(self, pretrained_model: nn.Module) -> None:
        super().__init__()
        self.model = pretrained_model

    def _raise_inference_error(self):
        raise NotImplementedError(
            f"{self.__class__.__name__} is solely used for OpenCatalyst leaderboard submissions; please call 'predict' from trainer."
        )

    def training_step(self, *args: Any, **kwargs: Any) -> None:
        self._raise_inference_error()

    def validation_step(self, *args: Any, **kwargs: Any) -> None:
        self._raise_inference_error()

    def test_step(self, *args: Any, **kwargs: Any) -> None:
        self._raise_inference_error()

    @abstractmethod
    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        ...


class IS2REInference(OpenCatalystInference):
    def __init__(
        self, pretrained_model: Union[AbstractEnergyModel, ScalarRegressionTask]
    ) -> None:
        assert isinstance(
            pretrained_model, (AbstractEnergyModel, ScalarRegressionTask)
        ), f"IS2REInference expects a pretrained energy model or 'ScalarRegressionTask' as input."
        super().__init__(pretrained_model)

    def forward(self, batch: BatchDict) -> DataDict:
        predictions = self.model(batch)
        return predictions


class S2EFInference(OpenCatalystInference):
    def __init__(self, pretrained_model: ForceRegressionTask) -> None:
        assert isinstance(
            pretrained_model, ForceRegressionTask
        ), f"S2EFInference expects a pretrained 'ForceRegressionTask' instance as input."
        super().__init__(pretrained_model)

    def forward(self, batch: BatchDict) -> DataDict:
        predictions = self.model(batch)
        return predictions

    def on_predict_start(self) -> None:
        self.apply(rnn_force_train_mode)
        return super().on_predict_start()

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        # force gradients when running predictions
        predictions = self(batch)
        energy, force = predictions["energy"], predictions["force"]
        energy = energy.detach().cpu().to(torch.float16)
        force = force.detach().cpu()
        ids, chunk_ids = batch.get("sid"), batch.get("fid")
        # ids are formatted differently for force tasks
        system_ids = [f"{i}_{j}" for i, j in zip(ids, chunk_ids)]
        predictions = {
            "ids": system_ids,
            "chunk_ids": chunk_ids,
            "energy": energy,
        }
        # processing the forces is a bit more complicated because apparently
        # only the free atoms are considered
        if self.regress_forces:
            if "graph" in batch:
                graph = batch.get("graph")
                fixed = graph.ndata["fixed"]
            else:
                # otherwise it's a point cloud
                fixed = batch.get("fixed")
            fixed_mask = fixed == 0
            # retrieve only forces corresponding to unfixed nodes
            predictions["forces"] = force[fixed_mask]
            natoms = tuple(batch.get("natoms").cpu().numpy().astype(int))
            chunk_split = torch.split(graph.ndata["fixed"], natoms)
            chunk_ids = []
            for chunk in chunk_split:
                ids = (len(chunk) - sum(chunk)).cpu().numpy().astype(int)
                chunk_ids.append(int(ids))

            predictions["chunk_ids"] = chunk_ids
        return predictions

    def on_predict_batch_end(
        self, outputs: Any, batch: Any, batch_idx: int, dataloader_idx: int
    ) -> None:
        # reset gradients to ensure no contamination between batches
        self.zero_grad(set_to_none=True)
