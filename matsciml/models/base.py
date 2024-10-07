# Copyright (C) 2022-3 Intel Corporation
# SPDX-License-Identifier: MIT License
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from contextlib import ExitStack, nullcontext
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, ContextManager, Dict, List, Optional, Type, Union
from warnings import warn
from logging import getLogger
from inspect import signature

import lightning.pytorch as pl
from lightning.pytorch import loggers as pl_loggers
import torch
from einops import reduce
from torch import Tensor, nn
from torch.optim import AdamW, Optimizer, lr_scheduler

from matsciml.common import package_registry
from matsciml.common.registry import registry
from matsciml.common.types import AbstractGraph, BatchDict, DataDict, Embeddings
from matsciml.models.common import OutputHead
from matsciml.modules.normalizer import Normalizer
from matsciml.models import losses as matsciml_losses

logger = getLogger("matsciml")
logger.setLevel("INFO")

if package_registry["dgl"]:
    import dgl

if package_registry["pyg"]:
    import torch_geometric as pyg

__all__ = [
    "AbstractEnergyModel",
    "ScalarRegressionTask",
    "BinaryClassificationTask",
    "ForceRegressionTask",
    "CrystalSymmetryClassificationTask",
    "MultiTaskLitModule",
    "OpenCatalystInference",
    "IS2REInference",
    "S2EFInference",
    "BaseTaskModule",
]

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


def prepend_affix(metrics: dict[str, torch.Tensor], affix: str) -> None:
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
        super().__init__()
        self.num_atoms = num_atoms
        self.bond_feat_dim = bond_feat_dim
        self.num_targets = num_targets

    def forward(self, data):
        raise NotImplementedError

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())


class AbstractTask(ABC, pl.LightningModule):
    # TODO the intention is for this class to supersede AbstractEnergyModel for DGL
    def __init__(
        self,
        atom_embedding_dim: int,
        num_atom_embedding: int = 100,
        embedding_kwargs: dict[str, Any] = {},
        encoder_only: bool = True,
    ) -> None:
        super().__init__()
        embedding_kwargs.setdefault("padding_idx", 0)
        self.atom_embedding = nn.Embedding(
            num_atom_embedding,
            atom_embedding_dim,
            **embedding_kwargs,
        )
        self.save_hyperparameters()

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @property
    def has_rnn(self) -> bool:
        """
        Returns True if any components of this model contains an RNN unit that
        inherits from 'nn.RNNBase'.
        """
        return any([isinstance(block, nn.RNNBase) for block in self.modules()])

    @abstractmethod
    def read_batch(self, batch: BatchDict) -> DataDict:
        """
        This method must be implemented by subclasses to extract
        input data out of a batch and into a dictionary format ready
        to be ingested by the actual model.

        Parameters
        ----------
        batch : BatchDict
            Batch of input data to be read

        Returns
        -------
        DataDict
            Dictionary containing input data, i.e. graphs and other
            tensor structures to be passed into the model
        """
        ...

    @abstractmethod
    def read_batch_size(self, batch: BatchDict) -> int | None: ...

    @abstractmethod
    def _forward(self, *args, **kwargs) -> Embeddings:
        """
        Implements the actual logic of the architecture. Given a set
        of input features, produce outputs/predictions from the model.

        Returns
        -------
        Embeddings
            Data structure containing system/graph and point/node level embeddings.
        """
        ...

    def forward(self, batch: BatchDict) -> Embeddings:
        """
        Given a batch structure, extract out data and pass it into the
        neural network architecture. This implements the 'forward' method
        as expected of all children of 'nn.Module'; it is not intended to
        be overridden, instead modify the 'read_batch' and '_forward' methods
        to change how this model/class of models interact with data.

        Parameters
        ----------
        batch : BatchDict
            Batch of data to process

        Returns
        -------
        Embeddings
            Data structure containing system/graph and point/node level embeddings.
        """
        input_data = self.read_batch(batch)
        outputs = self._forward(**input_data)
        # raise an error to help spot models that have not yet been refactored
        if not isinstance(outputs, Embeddings):
            raise ValueError(
                "Encoder did not return `Embeddings` data structure: please refactor your model!",
            )
        return outputs


class AbstractPointCloudModel(AbstractTask):
    def read_batch(self, batch: BatchDict) -> DataDict:
        r"""
        Extract data needed for point cloud modeling from a batch.

        Notably, to facilitate force calculation, the point cloud
        "neighborhood" for atom positions is constructed **after**
        giving the primary task (i.e. ``ForceRegressionTask``) an
        opportunity to enable gradients for each sample within the point cloud.

        To clarify usage of ``pos`` and ``pc_pos``, the former represents
        the packed batch of positions without separating them into their
        individual point clouds: **this is used for force computation**
        where we want to end up with a force tensor with the same shape.
        ``pc_pos`` corresponds to the padded, molecule centered point
        cloud data that should be used as input to a point cloud model.

        Parameters
        ----------
        batch : BatchDict
            Batch of samples to process

        Returns
        -------
        DataDict
            Input data for a point cloud model to process, notably
            including particle positions and features
        """
        from matsciml.datasets.utils import pad_point_cloud

        assert isinstance(
            batch["pos"],
            torch.Tensor,
        ), "Expect 'pos' data to be a packed tensor of shape [N, 3]"
        data = {key: batch.get(key) for key in ["pc_features", "pos"]}
        # split the stacked positions into each individual point cloud
        temp_pos = batch["pos"].split(batch["sizes"])
        pc_pos = []
        # sizes records the number of centers being used
        sizes = []
        # loop over each sample within a batch
        for index, sample in enumerate(temp_pos):
            src_nodes, dst_nodes = (
                batch["pc_src_nodes"][index],
                batch["pc_dst_nodes"][index],
            )
            # use dst_nodes to gauge size because you will always have more
            # dst nodes than src nodes right now
            sizes.append(len(dst_nodes))
            # carve out neighborhoods as dictated by the dataset/transform definition
            sample_pc_pos = (
                sample[src_nodes.long()][None, :] - sample[dst_nodes.long()][:, None]
            )
            pc_pos.append(sample_pc_pos)
        # pad the position result
        pc_pos, mask = pad_point_cloud(pc_pos, max(sizes))
        # get the features and make sure the shapes are consistent for the
        # batch and neighborhood
        feat_shape = data.get("pc_features").shape
        assert (
            pc_pos.shape[:-1] == feat_shape[:-1]
        ), "Shape of point cloud neighborhood positions is different from features!"
        data["pc_pos"] = pc_pos
        data["mask"] = mask
        data["sizes"] = sizes
        return data

    @abstractmethod
    def _forward(
        self,
        pc_pos: torch.Tensor,
        pc_features: torch.Tensor,
        mask: torch.Tensor | None = None,
        sizes: list[int] | None = None,
        **kwargs,
    ) -> Embeddings:
        """
        Sets expected patterns for args for point cloud based modeling, whereby
        the bare minimum expected data are 'pos' and 'pc_features' akin to graph
        approaches.

        Parameters
        ----------
        pc_pos : torch.Tensor
            Padded point cloud neighborhood tensor, with shape ``[B, N, M, 3]``
            for ``B`` batch size and ``N`` padded size. For full pairwise point
            clouds, ``N == M``.
        pc_features : torch.Tensor
            Padded point cloud feature tensor, with shape ``[B, N, M, D_in]``
            for ``B`` batch size and ``N`` padded size. For full pairwise point
            clouds, ``N == M``.
        mask : Optional[torch.Tensor], optional
            Boolean tensor with shape ``[B, N, M]``, by default None. If supplied
            in conjuction with ``sizes``, will mask out contributions from padding
            nodes.
        sizes : Optional[List[int]], optional
            List of integers denoting the size of the first non-batch point cloud
            dimension, by default None. If supplied in conjuction with ``mask``,
            will mask out contributions from padding nodes.

        Returns
        -------
        torch.Tensor
            Output of a point cloud model; system-level embedding or predictions
        """
        ...

    @staticmethod
    def mask_model_output(
        result: torch.Tensor,
        mask: torch.Tensor,
        sizes: list[int],
        extensive: bool,
    ) -> torch.Tensor:
        r"""
        Perform a masked reduction over a point cloud model output.

        This effectively removes the contributions from node centers or source
        particles, i.e. the first non-batch dimension, that correspond to padding nodes.
        The resulting shape should be ``[B, D]`` with ``B`` batch size and ``D``
        desired output dimension.

        Parameters
        ----------
        result : torch.Tensor
            Result of a point cloud model, with shape ``[B, N, M, D]``
            for ``B`` batch size, ``N`` padded source nodes, ``M``
            padded destination nodes, and output dimension ``D``.
        mask : torch.Tensor
            A 3D boolean tensor of shape ``[B, N, M]``
        sizes : List[int]
            A list comprising the number of atom centers that are not padding
            nodes.
        extensive : bool
            If ``True``, sums over nodes, otherwise performs a mean reduction.

        Returns
        -------
        torch.Tensor
            Per-point cloud results, with shape ``[B, D]``
        """
        # extract out a mask over [B, N] for N atom centers, removing
        # padded center node contributions to the system output
        center_mask = mask[..., 0]
        # this extracts a [N, D] tensor with N total particles, D embedding dim
        unpadded_result = result[center_mask]
        # this splits up into embeddings per node
        split_results = unpadded_result.split(sizes)
        # figure out what reduction to perform over the particles
        if extensive:
            reduce = torch.sum
        else:
            reduce = torch.mean
        # should be [B, D] for B systems
        output = torch.stack([reduce(t, dim=0) for t in split_results])
        return output

    def read_batch_size(self, batch: BatchDict) -> None:
        # returns None, because batch size can be readily determined by Lightning
        return None


class AbstractGraphModel(AbstractTask):
    def __init__(
        self,
        atom_embedding_dim: int,
        num_atom_embedding: int = 100,
        embedding_kwargs: dict[str, Any] = {},
        encoder_only: bool = True,
    ) -> None:
        super().__init__(
            atom_embedding_dim,
            num_atom_embedding,
            embedding_kwargs,
            encoder_only,
        )

    def read_batch(self, batch: BatchDict) -> DataDict:
        assert (
            "graph" in batch
        ), f"Model {self.__class__.__name__} expects graph structures, but 'graph' key was not found in batch."
        graph = batch.get("graph")
        return {"graph": graph}

    @staticmethod
    def join_position_embeddings(
        pos: torch.Tensor,
        node_feats: torch.Tensor,
    ) -> torch.Tensor:
        """
        This is a method for conveniently embedding both positions and node features
        together. Given that not every type of model will use this approach, it is
        left for concrete classes to utilize rather than being the default.

        Parameters
        ----------
        pos : torch.Tensor
            2D tensor with [N, 3] containing coordinates of each node in N
        node_feats : torch.Tensor
            2D tensor with [N, D] containing features of each node in N. Typically
            this pertains to the embedding lookup features, but up to the developer

        Returns
        -------
        torch.Tensor
            2D tensor with shape [N, D + 3]
        """
        return torch.hstack([pos, node_feats])

    @abstractmethod
    def _forward(
        self,
        graph: AbstractGraph,
        node_feats: torch.Tensor,
        pos: torch.Tensor | None = None,
        edge_feats: torch.Tensor | None = None,
        graph_feats: torch.Tensor | None = None,
        **kwargs,
    ) -> Embeddings:
        """
        Sets args/kwargs for the expected components of a graph-based
        model. At the bare minimum, we expect some kind of abstract
        graph structure, along with tensors of atomic coordinates and
        numbers to process. Optionally, models can include edge and graph
        features, but is left for concrete classes to implement how
        these are obtained.

        Parameters
        ----------
        graph : AbstractGraph
            Graph structure implemented in a particular framework
        node_feats : torch.Tensor
            Atomic numbers or other featurizations, typically shape [N, ...] for N nuclei
        pos : Optional[torch.Tensor]
            Atom positions with shape [N, 3], by default None to make this optional
            as some architectures may pass them as 'node_feats'
        edge_feats : Optional[torch.Tensor], optional
            Edge features to process, by default None
        graph_feats : Optional[torch.Tensor], optional
            Graph-level attributes/features to use, by default None

        Returns
        -------
        torch.Tensor
            Model output; either embedding or projected output
        """
        ...


if package_registry["dgl"]:

    class AbstractDGLModel(AbstractGraphModel):
        def read_batch(self, batch: BatchDict) -> DataDict:
            """
            Extract DGLGraph structure and features to pass into the model.

            More complicated models can override this method to extract out edge and
            graph features as well.

            Parameters
            ----------
            batch : BatchDict
                Batch of data to process.

            Returns
            -------
            DataDict
                Dictionary of input features to pass into the model
            """
            data = super().read_batch(batch)
            graph = data.get("graph")
            assert isinstance(
                graph,
                dgl.DGLGraph,
            ), f"Model {self.__class__.__name__} expects DGL graphs, but data in 'graph' key is type {type(graph)}"
            atomic_numbers = data["graph"].ndata["atomic_numbers"].long()
            node_embeddings = self.atom_embedding(atomic_numbers)
            pos = graph.ndata["pos"]
            # optionally can fuse into a single tensor with `self.join_position_embeddings`
            data["node_feats"] = node_embeddings
            data["pos"] = pos
            # these keys are left as None, but are filler for concrete models to extract
            data.setdefault("edge_feats", None)
            data.setdefault("graph_feats", None)
            return data

        def read_batch_size(self, batch: BatchDict) -> int:
            # grabs the number of batch samples from the DGLGraph attribute
            graph = batch["graph"]
            return graph.batch_size


if package_registry["pyg"]:

    class AbstractPyGModel(AbstractGraphModel):
        def read_batch(self, batch: BatchDict) -> DataDict:
            """
            Extract PyG structure and features to pass into the model.

            More complicated models can override this method to extract out edge and
            graph features as well.

            Parameters
            ----------
            batch : BatchDict
                Batch of data to process.

            Returns
            -------
            DataDict
                Dictionary of input features to pass into the model
            """
            data = super().read_batch(batch)
            graph = data.get("graph")
            assert isinstance(
                graph,
                (pyg.data.Data, pyg.data.Batch),
            ), f"Model {self.__class__.__name__} expects PyG graphs, but data in 'graph' key is type {type(graph)}"
            for key in ["edge_feats", "graph_feats"]:
                data[key] = getattr(graph, key, None)
            atomic_numbers: torch.Tensor = getattr(graph, "atomic_numbers").to(
                torch.int,
            )
            node_embeddings = self.atom_embedding(atomic_numbers)
            pos: torch.Tensor = getattr(graph, "pos")
            # optionally can fuse into a single tensor with `self.join_position_embeddings`
            data["node_feats"] = node_embeddings
            data["pos"] = pos
            return data

        def read_batch_size(self, batch: BatchDict) -> int:
            graph = batch["graph"]
            return graph.num_graphs


class AbstractEnergyModel(pl.LightningModule):
    """
    At a minimum, the point of this is to help register associated models
    with PyTorch Lightning ModelRegistry; the expectation is that you get
    the graph energy as well as the atom forces.

    TODO - replace this class with `AbstractTask`, see #167 and #168
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


@registry.register_task("BaseTaskModule")
class BaseTaskModule(pl.LightningModule):
    __task__ = None
    __needs_grads__ = []

    def __init__(
        self,
        encoder: nn.Module | None = None,
        encoder_class: type[nn.Module] | None = None,
        encoder_kwargs: dict[str, Any] | None = None,
        loss_func: (
            type[nn.Module] | nn.Module | dict[str, nn.Module | type[nn.Module]] | None
        ) = None,
        task_keys: list[str] | None = None,
        output_kwargs: dict[str, Any] = {},
        lr: float = 1e-4,
        weight_decay: float = 0.0,
        embedding_reduction_type: str = "mean",
        normalize_kwargs: dict[str, float] | None = None,
        scheduler_kwargs: dict[str, dict[str, Any]] | None = None,
        log_embeddings: bool = False,
        log_embeddings_every_n_steps: int = 50,
        **kwargs,
    ) -> None:
        super().__init__()
        if encoder is not None:
            warn(
                f"Encoder object was passed directly into {self.__class__.__name__}; saved hyperparameters will be incomplete!",
            )
        if encoder_class is not None and encoder_kwargs:
            try:
                encoder = encoder_class(**encoder_kwargs)
            except:  # noqa: E722
                raise ValueError(
                    f"Unable to instantiate encoder {encoder_class} with kwargs: {encoder_kwargs}.",
                )
        if encoder is not None:
            self.encoder = encoder
        else:
            raise ValueError("No valid encoder passed.")
        if isinstance(loss_func, type):
            loss_func = loss_func()
        # if we have a dictionary mapping, we specify the loss function
        # for each target
        if isinstance(loss_func, dict):
            for key in loss_func:
                # initialize objects if types are provided
                if isinstance(loss_func[key], type):
                    loss_func[key] = loss_func[key]()
                if task_keys and key not in task_keys:
                    raise KeyError(
                        f"Loss dict configured with {key}/{loss_func[key]} that was not provided in task keys."
                    )
            if not task_keys:
                logger.warning(
                    f"Task keys were not specified, using loss_func keys instead {loss_func.keys()}"
                )
                task_keys = list(loss_func.keys())
            # convert to a module dict for consistent API usage
            loss_func = nn.ModuleDict({key: value for key, value in loss_func.items()})
        self.loss_func = loss_func
        default_heads = {"act_last": None, "hidden_dim": 128}
        default_heads.update(output_kwargs)
        self.output_kwargs = default_heads
        self.normalize_kwargs = normalize_kwargs
        self.task_keys = task_keys
        self._task_loss_scaling = kwargs.get("task_loss_scaling", {})
        if len(self.task_keys) > 0:
            self.task_loss_scaling = self._task_loss_scaling
        self.embedding_reduction_type = embedding_reduction_type
        self.save_hyperparameters(ignore=["encoder", "loss_func"])

    @property
    def task_keys(self) -> list[str]:
        return self._task_keys

    @task_keys.setter
    def task_keys(self, values: set | list[str] | None) -> None:
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
        # homogenize it into a dictionary mapping
        if isinstance(self.loss_func, nn.Module) and not isinstance(
            self.loss_func, nn.ModuleDict
        ):
            loss_dict = nn.ModuleDict({key: deepcopy(self.loss_func) for key in values})
            self.loss_func = loss_dict
        # if a task key was given but not contained in loss_func
        # user needs to figure out what to do
        for key in values:
            if key not in self.loss_func.keys():
                raise KeyError(
                    f"Task key {key} was specified but no loss function was specified."
                )
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

    @property
    def task_loss_scaling(self) -> dict[str, float]:
        return self._task_loss_scaling

    @task_loss_scaling.setter
    def task_loss_scaling(self, loss_scaling_dict: dict[str, float] | None) -> None:
        """
        Adds loss scaling per task. Ensures task loss scaling key is present in the
        task loss keys.

        Parameters
        ----------
        loss_scaling_dict : dict[str, float]
            Dictionary used to map a task key to a scaling value.
        """
        if not isinstance(loss_scaling_dict, dict):
            raise TypeError(
                f"task_loss_scaling {loss_scaling_dict} must be a dictionary, got type {type(loss_scaling_dict)}."
            )

        keys = set(loss_scaling_dict.keys()).union(set(self._task_keys))
        not_in_scaling = []
        not_in_tasks = []
        final_scaling_dict = {}
        for key in keys:
            if key not in loss_scaling_dict:
                not_in_scaling.append(key)
            if key not in self._task_keys:
                not_in_tasks.append(keys)
            final_scaling_dict[key] = loss_scaling_dict.get(key, 1.0)

        if len(not_in_scaling) != 0:
            warn(f"Unspecified tasks will have a scaling of 1: {not_in_scaling}")
        if len(not_in_tasks) != 0:
            warn(
                f"Task scaling keys passed that are not actually tasks: {not_in_tasks}"
            )

        self._task_loss_scaling = final_scaling_dict
        self.hparams["task_loss_scaling"] = self._task_loss_scaling

    @abstractmethod
    def _make_output_heads(self) -> nn.ModuleDict: ...

    @property
    def output_heads(self) -> nn.ModuleDict:
        return self._output_heads

    @output_heads.setter
    def output_heads(self, heads: nn.ModuleDict) -> None:
        assert isinstance(
            heads,
            nn.ModuleDict,
        ), "Output heads must be an instance of `nn.ModuleDict`."
        assert len(heads) > 0, f"No output heads in {heads}."
        assert all(
            [key in self.task_keys for key in heads.keys()],
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
        batch: dict[str, torch.Tensor | dgl.DGLGraph | dict[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        embeddings = self.encoder(batch)
        batch["embeddings"] = embeddings
        outputs = self.process_embedding(embeddings)
        return outputs

    def process_embedding(self, embeddings: Embeddings) -> dict[str, torch.Tensor]:
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
            # in the event that we get multiple embeddings, we average
            # every dimension execpt the batch and dimensionality
            output = head(embeddings.system_embedding)
            output = reduce(
                output,
                "b ... d -> b d",
                reduction=self.embedding_reduction_type,
            )
            results[key] = output
        return results

    def _log_embedding(self, embeddings: Embeddings) -> None:
        """
        This maps the appropriate logging function depending on what
        logger was used, and saves the graph and node level embeddings.

        Some services like ``wandb`` are able to do some nifty embedding
        analyses online using these embeddings.

        Parameters
        ----------
        embeddings : Embeddings
            Data structure containing embeddings from the encoder.
        """
        log_freq = self.hparams.log_embeddings_every_n_steps
        global_step = self.trainer.global_step
        # only log embeddings at the same cadence as everything else
        if self.logger is not None and (global_step % log_freq) == 0:
            exp = self.logger.experiment
            sys_z = embeddings.system_embedding.detach().cpu()
            node_z = embeddings.point_embedding.detach().cpu()
            if isinstance(self.logger, pl_loggers.WandbLogger):
                # this import is okay here since we need it for the logger anyway
                import wandb

                cols = [f"D{i}" for i in range(sys_z.size(-1))]
                exp.log(
                    {"graph_embeddings": wandb.Table(columns=cols, data=sys_z.tolist())}
                )
                if isinstance(embeddings.point_embedding, torch.Tensor):
                    # TODO: should add labels to the nodes based on graph index
                    exp.log(
                        {
                            "node_embeddings": wandb.Table(
                                columns=cols, data=node_z.tolist()
                            )
                        }
                    )
            elif isinstance(self.logger, pl_loggers.TensorBoardLogger):
                exp.add_embedding(
                    sys_z,
                    tag=f"graph_embeddings_{self.trainer.global_step}",
                )
                if isinstance(embeddings.point_embedding, torch.Tensor):
                    # TODO: should add labels to the nodes based on graph index
                    exp.add_embedding(
                        node_z,
                        tag=f"node_embeddings_{self.trainer.global_step}",
                    )
            else:
                pass

    def _get_targets(
        self,
        batch: dict[str, torch.Tensor | dgl.DGLGraph | dict[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
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
        assert len(self.task_keys) != 0, "No target keys were set!"
        for key in self.task_keys:
            target_dict[key] = batch["targets"][key]
        return target_dict

    def _filter_task_keys(
        self,
        keys: list[str],
        batch: dict[str, torch.Tensor | dgl.DGLGraph | dict[str, torch.Tensor]],
    ) -> list[str]:
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
        batch: dict[str, torch.Tensor | dgl.DGLGraph | dict[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
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
        # if we have EMA weights, use them for prediction instead
        if hasattr(self, "ema_module") and not self.training:
            wrapper = self.ema_module
        else:
            wrapper = self
        predictions = wrapper(batch)
        losses = {}
        for key in self.task_keys:
            target_val = targets[key]
            if self.uses_normalizers:
                target_val = self.normalizers[key].norm(target_val)
            loss_func = self.loss_func[key]
            # determine if we need additional arguments
            loss_func_signature = signature(loss_func.forward).parameters
            kwargs = {"input": predictions[key], "target": target_val}
            # pack atoms per graph information too
            if "atoms_per_graph" in loss_func_signature:
                if graph := batch.get("graph", None):
                    if isinstance(graph, dgl.DGLGraph):
                        num_atoms = graph.batch_num_nodes()
                    else:
                        # in the pyg case we use the pointer tensor
                        num_atoms = graph.ptr[1:] - graph.ptr[:-1]
                else:
                    # in MP at least this is provided by the dataset class
                    num_atoms = batch.get("sizes", None)
                    if not num_atoms:
                        raise NotImplementedError(
                            "Unable to determine number of atoms for dataset. "
                            "This is required for the atom-weighted loss functions."
                        )
                kwargs["atoms_per_graph"] = num_atoms
            loss = loss_func(**kwargs)
            losses[key] = loss * self.task_loss_scaling[key]

        total_loss: torch.Tensor = sum(losses.values())
        return {"loss": total_loss, "log": losses}

    def configure_optimizers(self) -> torch.optim.AdamW:
        opt = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        # configure schedulers as a nested dictionary
        schedule_dict = getattr(self.hparams, "scheduler_kwargs", None)
        schedulers = []
        if schedule_dict:
            for scheduler_name, params in schedule_dict.items():
                # try get the scheduler class
                scheduler_class = getattr(lr_scheduler, scheduler_name, None)
                if not scheduler_class:
                    raise NameError(
                        f"{scheduler_class} was requested for LR scheduling, but is not in 'torch.optim.lr_scheduler'.",
                    )
                scheduler = scheduler_class(opt, **params)
                schedulers.append(scheduler)
        return [opt], schedulers

    def training_step(
        self,
        batch: dict[str, torch.Tensor | dgl.DGLGraph | dict[str, torch.Tensor]],
        batch_idx: int,
    ):
        loss_dict = self._compute_losses(batch)
        metrics = {}
        # prepending training flag for
        for key, value in loss_dict["log"].items():
            metrics[f"train_{key}"] = value
        try:
            batch_size = self.encoder.read_batch_size(batch)
        except:  # noqa: E722
            warn(
                "Unable to parse batch size from data, defaulting to `None` for logging.",
            )
            batch_size = None
        self.log_dict(metrics, on_step=True, prog_bar=True, batch_size=batch_size)
        if self.hparams.log_embeddings and "embeddings" in batch:
            self._log_embedding(batch["embeddings"])
        return loss_dict

    def validation_step(
        self,
        batch: dict[str, torch.Tensor | dgl.DGLGraph | dict[str, torch.Tensor]],
        batch_idx: int,
    ):
        loss_dict = self._compute_losses(batch)
        metrics = {}
        # prepending training flag for
        for key, value in loss_dict["log"].items():
            metrics[f"val_{key}"] = value
        try:
            batch_size = self.encoder.read_batch_size(batch)
        except:  # noqa: E722
            warn(
                "Unable to parse batch size from data, defaulting to `None` for logging.",
            )
            batch_size = None
        self.log_dict(metrics, batch_size=batch_size, on_step=True, prog_bar=True)
        if self.hparams.log_embeddings and "embeddings" in batch:
            self._log_embedding(batch["embeddings"])
        return loss_dict

    def test_step(
        self,
        batch: dict[str, torch.Tensor | dgl.DGLGraph | dict[str, torch.Tensor]],
        batch_idx: int,
    ):
        loss_dict = self._compute_losses(batch)
        metrics = {}
        # prepending training flag for
        for key, value in loss_dict["log"].items():
            metrics[f"test_{key}"] = value
        try:
            batch_size = self.encoder.read_batch_size(batch)
        except:  # noqa: E722
            warn(
                "Unable to parse batch size from data, defaulting to `None` for logging.",
            )
            batch_size = None
        self.log_dict(metrics, batch_size=batch_size, on_epoch=True, prog_bar=True)
        if self.hparams.log_embeddings and "embeddings" in batch:
            self._log_embedding(batch["embeddings"])
        return loss_dict

    def _make_normalizers(self) -> dict[str, Normalizer]:
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

    def predict(self, batch: BatchDict) -> dict[str, torch.Tensor]:
        """
        Implements what is effectively the 'inference' logic of the task,
        where run the forward pass on a batch of samples, and if normalizers
        were used for training, we also apply the inverse operation to get
        values in the right scale.

        Not to be confused with `predict_step`, which is used by Lightning as
        part of the prediction workflow. Since there is no one-size-fits-all
        inference workflow we can define, this provides a convenient function
        for users to call as a replacement.

        Parameters
        ----------
        batch : BatchDict
            Batch of samples to pass to the model.

        Returns
        -------
        dict[str, torch.Tensor]
            Output dictionary as provided by the forward pass, but if
            normalizers are available for a given task, we apply the
            inverse norm on the value.
        """
        # use EMA weights instead if they are available
        if hasattr(self, "ema_module"):
            wrapper = self.ema_module
        else:
            wrapper = self
        outputs = wrapper(batch)
        if self.uses_normalizers:
            for key in self.task_keys:
                if key in self.normalizers:
                    # apply the inverse transform if provided
                    outputs[key] = self.normalizers[key].denorm(outputs[key])
        return outputs

    @classmethod
    def from_pretrained_encoder(cls, task_ckpt_path: str | Path, **kwargs):
        """
        Attempts to instantiate a new task, adopting a previously trained encoder model.

        This function will load in a saved PyTorch Lightning checkpoint,
        copy over the hyperparameters needed to reconstruct the encoder,
        and simply maps the encoder ``state_dict`` to the new instance.

        ``Kwargs`` are passed directly into the creation of the task, and so can
        be thought of as just a task through the typical interface normally.

        Parameters
        ----------
        task_ckpt_path : Union[str, Path]
            Path to an existing task checkpoint file. Typically, this
            would be a PyTorch Lightning checkpoint.

        Examples
        --------
        1. Create a new task simply from training another one

        >>> new_task = ScalarRegressionTask.from_pretrained_encoder(
            "epoch=10-step=100.ckpt"
            )

        2. Create a new task, modifying output heads

        >>> new_taks = ForceRegressionTask.from_pretrained_encoder(
            "epoch=5-step=12516.ckpt",
            output_kwargs={
                "num_hidden": 3,
                "activation": "nn.ReLU"
            }
        )
        """
        if isinstance(task_ckpt_path, str):
            task_ckpt_path = Path(task_ckpt_path)
        assert (
            task_ckpt_path.exists()
        ), "Encoder checkpoint filepath specified but does not exist."
        ckpt = torch.load(task_ckpt_path)
        for key in ["encoder_class", "encoder_kwargs"]:
            assert (
                key in ckpt["hyper_parameters"]
            ), f"{key} expected to be in hyperparameters, but was not found."
            # copy over the data for the new task
            kwargs[key] = ckpt["hyper_parameters"][key]
        # construct the new task with random weights
        task = cls(**kwargs)
        # this only copies over encoder weights, and removes the 'encoder.'
        # pattern from keys
        encoder_weights = {
            key.replace("encoder.", ""): tensor
            for key, tensor in ckpt["state_dict"].items()
            if "encoder." in key
        }
        # load in pre-trained weights
        task.encoder.load_state_dict(encoder_weights)
        return task

    def on_save_checkpoint(self, checkpoint):
        """Override checkpoint saving to facilitate EMA weights loading."""
        super().on_save_checkpoint(checkpoint)
        # this case checks for EMA weights, where we give up on
        # the non-averaged weights in favor of the averaged ones
        checkpoint["state_dict"] = self.state_dict()
        if hasattr(self, "ema_module"):
            logger.info("Replacing model weights with exponential moving average.")
            for key, value in self.ema_module.state_dict().items():
                if key != "n_averaged":
                    # remove the module prefix for EMA state
                    tree = key.split(".")
                    renamed_key = ".".join(tree[1:])
                    assert (
                        renamed_key in checkpoint["state_dict"]
                    ), f"Unexpected key in EMA module: {renamed_key}"
                    checkpoint["state_dict"][renamed_key] = value
            # now remove the redundant EMA weights to unblock checkpoint loading later
            for key in list(checkpoint["state_dict"].keys()):
                if "ema_module" in key:
                    del checkpoint["state_dict"][key]


@registry.register_task("ScalarRegressionTask")
class ScalarRegressionTask(BaseTaskModule):
    __task__ = "regression"

    """
    NOTE: You can have multiple targets, but each target is scalar.
    """

    def __init__(
        self,
        encoder: nn.Module | None = None,
        encoder_class: type[nn.Module] | None = None,
        encoder_kwargs: dict[str, Any] | None = None,
        loss_func: (
            type[nn.Module] | nn.Module | dict[str, nn.Module | type[nn.Module]] | None
        ) = nn.MSELoss,
        task_keys: list[str] | None = None,
        output_kwargs: dict[str, Any] = {},
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
            modules[key] = OutputHead(1, **self.output_kwargs).to(
                self.device, dtype=self.dtype
            )
        return nn.ModuleDict(modules)

    def _filter_task_keys(
        self,
        keys: list[str],
        batch: dict[str, torch.Tensor | dgl.DGLGraph | dict[str, torch.Tensor]],
    ) -> list[str]:
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

    def on_train_batch_start(self, batch: Any, batch_idx: int) -> int | None:
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
            self.task_loss_scaling = self._task_loss_scaling

        return status

    def on_validation_batch_start(
        self,
        batch: any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        self.on_train_batch_start(batch, batch_idx)


@registry.register_task("MaceEnergyForceTask")
class MaceEnergyForceTask(BaseTaskModule):
    __task__ = "regression"
    """
    Class for training MACE on energy and forces

    """

    def __init__(
        self,
        encoder: Optional[nn.Module] = None,
        encoder_class: Optional[Type[nn.Module]] = None,
        encoder_kwargs: Optional[Dict[str, Any]] = None,
        loss_func: (
            type[nn.Module] | nn.Module | dict[str, nn.Module | type[nn.Module]] | None
        ) = nn.MSELoss,
        loss_coeff: Optional[Dict[str, Any]] = None,
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
        self.loss_coeff = loss_coeff

    def process_embedding(self, embeddings: Embeddings) -> Dict[str, torch.Tensor]:
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
            # in the event that we get multiple embeddings, we average
            # every dimension execpt the batch and dimensionality
            output = head(embeddings.system_embedding[key])
            output = reduce(output, "b ... d -> b d", reduction="mean")
            results[key] = output
        if self.hparams.log_embeddings:
            self._log_embedding(embeddings)
        return results

    def _compute_losses(
        self,
        batch: Dict[str, Union[torch.Tensor, dgl.DGLGraph, Dict[str, torch.Tensor]]],
    ) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Compute pred versus target for every target, then sum.
        With coefficients defined for each key
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
            if self.loss_coeff is None:
                coefficient = 1.0
            else:
                coefficient = self.loss_coeff[key]

            losses[key] = self.loss_func(predictions[key], target_val) * (
                coefficient / predictions[key].numel()
            )
        total_loss: torch.Tensor = sum(losses.values())
        return {"loss": total_loss, "log": losses}

    def _make_output_heads(self) -> nn.ModuleDict:
        modules = {}
        for key in self.task_keys:
            modules[key] = OutputHead(**self.output_kwargs[key]).to(
                self.device, dtype=self.dtype
            )
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

    def validation_step(
        self,
        batch: Dict[str, Union[torch.Tensor, dgl.DGLGraph, Dict[str, torch.Tensor]]],
        batch_idx: int,
    ):
        with torch.enable_grad():  # Enabled gradient for Force computation
            loss_dict = self._compute_losses(batch)
        metrics = {}
        # prepending training flag for
        for key, value in loss_dict["log"].items():
            metrics[f"val_{key}"] = value
        try:
            batch_size = self.encoder.read_batch_size(batch)
        except:  # noqa: E722
            warn(
                "Unable to parse batch size from data, defaulting to `None` for logging."
            )
            batch_size = None
        self.log_dict(metrics, batch_size=batch_size, on_step=True, prog_bar=True)
        return loss_dict

    def test_step(
        self,
        batch: Dict[str, Union[torch.Tensor, dgl.DGLGraph, Dict[str, torch.Tensor]]],
        batch_idx: int,
    ):
        with torch.enable_grad():  # Enabled gradient for Force computation
            loss_dict = self._compute_losses(batch)
        metrics = {}
        # prepending training flag for
        for key, value in loss_dict["log"].items():
            metrics[f"test_{key}"] = value
        try:
            batch_size = self.encoder.read_batch_size(batch)
        except:  # noqa: E722
            warn(
                "Unable to parse batch size from data, defaulting to `None` for logging."
            )
            batch_size = None
        self.log_dict(metrics, batch_size=batch_size, on_step=True, prog_bar=True)
        return loss_dict

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
        self, batch: any, batch_idx: int, dataloader_idx: int = 0
    ):
        self.on_train_batch_start(batch, batch_idx)


@registry.register_task("BinaryClassificationTask")
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
        encoder: nn.Module | None = None,
        encoder_class: type[nn.Module] | None = None,
        encoder_kwargs: dict[str, Any] | None = None,
        loss_func: (
            type[nn.Module] | nn.Module | dict[str, nn.Module | type[nn.Module]] | None
        ) = nn.BCEWithLogitsLoss,
        task_keys: list[str] | None = None,
        output_kwargs: dict[str, Any] = {},
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
            modules[key] = OutputHead(1, **self.output_kwargs).to(
                self.device, dtype=self.dtype
            )
        return nn.ModuleDict(modules)

    def on_train_batch_start(self, batch: Any, batch_idx: int) -> int | None:
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
            keys = batch["target_types"]["classification"]
            self.task_keys = keys
            # now add the parameters to our task's optimizer
            opt = self.optimizers()
            opt.add_param_group({"params": self.output_heads.parameters()})
            self.task_loss_scaling = self._task_loss_scaling

        return status

    def on_validation_batch_start(
        self,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        self.on_train_batch_start(batch, batch_idx)


@registry.register_task("ForceRegressionTask")
class ForceRegressionTask(BaseTaskModule):
    __task__ = "regression"
    __needs_grads__ = ["pos"]

    def __init__(
        self,
        encoder: nn.Module | None = None,
        encoder_class: type[nn.Module] | None = None,
        encoder_kwargs: dict[str, Any] | None = None,
        loss_func: (
            type[nn.Module] | nn.Module | dict[str, nn.Module | type[nn.Module]] | None
        ) = None,
        task_keys: list[str] | None = None,
        output_kwargs: dict[str, Any] = {},
        embedding_reduction_type: str = "sum",
        compute_stress: bool = False,
        **kwargs,
    ) -> None:
        if not loss_func:
            logger.warning(
                "Loss functions were not specified. "
                "Defaulting to AtomWeightedMSE for energy and MSE for force."
            )
            loss_func = {
                "energy": matsciml_losses.AtomWeightedMSE(),
                "force": nn.MSELoss(),
            }
        super().__init__(
            encoder,
            encoder_class,
            encoder_kwargs,
            loss_func,
            task_keys,
            output_kwargs,
            embedding_reduction_type=embedding_reduction_type,
            **kwargs,
        )
        self.save_hyperparameters(ignore=["encoder", "loss_func"])
        # have to enable double backprop
        self.automatic_optimization = False
        self.compute_stress = compute_stress

    def _make_output_heads(self) -> nn.ModuleDict:
        # this task only utilizes one output head
        modules = {
            "energy": OutputHead(1, **self.output_kwargs).to(
                self.device, dtype=self.dtype
            )
        }
        return nn.ModuleDict(modules)

    def forward(
        self,
        batch: dict[str, torch.Tensor | dgl.DGLGraph | dict[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        # virial/stress computation inspired from https://github.com/ACEsuit/mace/blob/main/mace/modules/utils.py
        # for ease of use, this task will always compute forces
        with dynamic_gradients_context(True, self.has_rnn):
            # first ensure that positions tensor is backprop ready
            if "graph" in batch:
                graph = batch["graph"]
                # the DGL case
                if hasattr(graph, "ndata"):
                    pos: torch.Tensor = graph.ndata.get("pos")
                    # for frame averaging
                    fa_rot = graph.ndata.get("fa_rot", None)
                    fa_pos = graph.ndata.get("fa_pos", None)
                else:
                    # otherwise assume it's PyG
                    pos: torch.Tensor = graph.pos
                    # for frame averaging
                    fa_rot = getattr(graph, "fa_rot", None)
                    fa_pos = getattr(graph, "fa_pos", None)
            else:
                graph = None
                # assume point cloud otherwise
                pos: torch.Tensor = batch.get("pos")
                # no frame averaging architecture yet for point clouds
                fa_rot = None
                fa_pos = None

            if self.compute_stress:
                num_graphs = batch["graph"].batch_size
                src_nodes = torch.cat(batch["src_nodes"])
                cell = batch.get("cell")
                unit_offsets = batch.get("unit_offsets")
                batch_indices = torch.concat(
                    [
                        torch.tensor([idx] * batch["natoms"][idx].long())
                        for idx in range(num_graphs)
                    ]
                )
                pos, shifts, displacement, symmetric_displacement = (
                    self.get_symmetric_displacement(
                        pos, unit_offsets, cell, src_nodes, num_graphs, batch_indices
                    )
                )
                batch["displacement"] = displacement

                if "graph" in batch:
                    graph.pos = pos
                if hasattr(graph, "ndata"):
                    graph.ndata["pos"] = pos

                if fa_pos is not None:
                    for index in range(len(fa_pos)):
                        fa_pos[index].requires_grad_(True)
                        fa_pos[index] = fa_pos[index] + torch.einsum(
                            "be,bec->bc",
                            fa_pos[index],
                            symmetric_displacement[batch_indices],
                        )

            if pos is None:
                raise ValueError(
                    "No atomic positions were found in batch - neither as standalone tensor nor graph.",
                )
            if isinstance(pos, torch.Tensor):
                pos.requires_grad_(True)
            elif isinstance(pos, list):
                [p.requires_grad_(True) for p in pos]
            else:
                raise ValueError(
                    f"'pos' data is required for force calculation, but isn't a tensor or a list of tensors: {type(pos)}.",
                )
            if isinstance(fa_pos, torch.Tensor):
                fa_pos.requires_grad_(True)
            elif isinstance(fa_pos, list):
                [f_p.requires_grad_(True) for f_p in fa_pos]
            if "embeddings" in batch:
                embeddings = batch.get("embeddings")
            else:
                embeddings = self.encoder(batch)

            outputs = self.process_embedding(embeddings, batch, pos, fa_rot, fa_pos)
        return outputs

    def process_embedding(
        self,
        embeddings: Embeddings,
        batch: BatchDict,
        pos: torch.Tensor,
        fa_rot: None | torch.Tensor = None,
        fa_pos: None | torch.Tensor = None,
    ) -> dict[str, torch.Tensor]:
        graph = batch.get("graph")
        natoms = batch.get("natoms")

        outputs = {}
        # compute node-level contributions to the energy
        node_energies = self.output_heads["energy"](embeddings.point_embedding)
        if graph is not None:
            if isinstance(graph, dgl.DGLGraph):
                graph.ndata["node_energies"] = node_energies

                def readout(node_energies: torch.Tensor):
                    return dgl.readout_nodes(
                        graph, "node_energies", op=self.embedding_reduction_type
                    )

            else:
                # assumes a batched pyg graph
                batch_indices = getattr(graph, "batch", None)
                if batch_indices is None:
                    batch_indices = torch.zeros_like(graph.atomic_numbers)
                from torch_geometric.utils import scatter

                def readout(node_energies: torch.Tensor):
                    return scatter(
                        node_energies,
                        batch_indices,
                        dim=-2,
                        reduce=self.embedding_reduction_type,
                    )

        else:

            def readout(node_energies: torch.Tensor):
                return reduce(
                    node_energies, "b ... d -> b ()", self.embedding_reduction_type
                )

        def energy_and_force(
            pos: torch.Tensor, node_energies: torch.Tensor, readout: Callable
        ) -> tuple[torch.Tensor, torch.Tensor]:
            # we sum over points and keep dimension as 1
            energy = readout(node_energies)
            if energy.ndim == 1:
                energy.unsqueeze(-1)
            # now use autograd for force and virials calculation
            inputs = [pos]

            if self.compute_stress:
                displacement = batch["displacement"]
                inputs.append(displacement)

            # outputs will be (force, virials)
            outputs = torch.autograd.grad(
                outputs=energy,
                inputs=inputs,
                grad_outputs=[torch.ones_like(energy)],
                create_graph=True,
            )

            if self.compute_stress:
                virials = outputs[1]
                stress = torch.zeros_like(displacement)
                cell = batch["cell"].view(-1, 3, 3)
                volume = torch.linalg.det(cell).abs().unsqueeze(-1)
                stress = virials / volume.view(-1, 1, 1)
                stress = torch.where(
                    torch.abs(stress) < 1e10, stress, torch.zeros_like(stress)
                )
                stress = -1 * stress
            else:
                stress = None
                virials = None
            force = -1 * outputs[0]
            return energy, force, stress, virials

        # not using frame averaging
        if fa_pos is None:
            energy, force, stress, virials = energy_and_force(
                pos, node_energies, readout
            )
        else:
            energy = []
            force = []
            stress = []
            virials = []
            for idx, pos in enumerate(fa_pos):
                frame_embedding = node_energies[:, idx, :]
                frame_energy, frame_force, frame_stress, frame_virials = (
                    energy_and_force(pos, frame_embedding, readout)
                )
                force.append(frame_force)
                energy.append(frame_energy.unsqueeze(-1))
                stress.append(frame_stress)
                virials.append(frame_virials)

        # check to see if we are frame averaging
        if fa_rot is not None:
            all_forces = []
            all_stress = []
            all_virials = []
            # loop over each frame prediction, and transform to guarantee
            # equivariance of frame averaging method
            natoms = natoms.squeeze(-1).to(int)
            for frame_idx, frame_rot in enumerate(fa_rot):
                force_repeat_rot = torch.repeat_interleave(
                    frame_rot,
                    natoms,
                    dim=0,
                ).to(self.device)
                rotated_forces = (
                    force[frame_idx]
                    .view(-1, 1, 3)
                    .bmm(force_repeat_rot.transpose(1, 2))
                )
                all_forces.append(rotated_forces)
            if self.compute_stress:
                for frame_idx, frame_rot in enumerate(fa_rot):
                    rotated_stress = stress[frame_idx].bmm(frame_rot.transpose(1, 2))
                    rotated_virials = virials[frame_idx].bmm(frame_rot.transpose(1, 2))
                    # adding dim to concat on
                    all_stress.append(rotated_stress.unsqueeze(1))
                    all_virials.append(rotated_virials.unsqueeze(1))

            # combine all the force and energy data into a single tensor
            # using frame averaging, the expected shapes after concatenation are:
            # force - [num positions, num frames, 3]
            # energy - [batch size, num frames, 1]
            # stress/virials - [batch size, 3, 3]
            force = torch.cat(all_forces, dim=1)
            energy = torch.cat(energy, dim=1)
            if self.compute_stress:
                stress = torch.cat(all_stress, dim=1)
                virials = torch.cat(all_virials, dim=1)
        # reduce outputs to what are expected shapes
        outputs["force"] = reduce(
            force,
            "n ... d -> n d",
            self.embedding_reduction_type,
            d=3,
        )
        if self.compute_stress:
            outputs["stress"] = reduce(
                stress,
                "b ... n d -> b n d",
                self.embedding_reduction_type,
                d=3,
            )
            outputs["virials"] = reduce(
                virials,
                "b ... n d -> b n d",
                self.embedding_reduction_type,
                d=3,
            )
        # this may not do anything if we aren't frame averaging
        # since the reduction is also done in the energy_and_force call
        outputs["energy"] = reduce(
            energy,
            "b ... d -> b d",
            self.embedding_reduction_type,
            d=1,
        )
        # this ensures that we get a scalar value for every node
        # representing the energy contribution
        outputs["node_energies"] = node_energies
        if self.hparams.log_embeddings:
            self._log_embedding(embeddings)
        return outputs

    def predict(self, batch: BatchDict) -> dict[str, torch.Tensor]:
        """
        Similar to the base method, but we make two minor modifications to
        the denormalization logic as we want to potentially apply the same
        energy normalization rescaling to the forces and node-level energies.

        Parameters
        ----------
        batch : BatchDict
            Batch of samples to evaluate on.

        Returns
        -------
        dict[str, torch.Tensor]
            Output dictionary as provided by the forward call. For this task in
            particular, we may also apply the energy rescaling to forces and
            node energies if separate keys for them are not provided.
        """
        output = super().predict(batch)
        # for forces, in the event that a dedicated normalizer wasn't provided
        # but we have an energy normalizer, we apply the same factors to the force
        if self.uses_normalizers:
            if "force" not in self.normalizers and "energy" in self.normalizers:
                # for force only std is used to rescale
                output["force"] = output["force"] * self.normalizers["energy"].std
            if "node_energies" not in self.normalizers and "energy" in self.normalizers:
                output["node_energies"] = self.normalizers["energy"].denorm(
                    output["node_energies"]
                )
        return output

    def _get_targets(
        self,
        batch: dict[str, torch.Tensor | dgl.DGLGraph | dict[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
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
                    f"{key} was not found in targets key in batch, which is needed for force regression task.",
                ) from e
        return target_dict

    def on_train_batch_start(self, batch: Any, batch_idx: int) -> int | None:
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
            self.task_loss_scaling = self._task_loss_scaling

        return status

    def training_step(
        self,
        batch: dict[str, torch.Tensor | dgl.DGLGraph | dict[str, torch.Tensor]],
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
        self.manual_backward(loss)
        self.on_before_optimizer_step(opt)
        opt.step()
        metrics = {}
        # prepending training flag
        for key, value in loss_dict["log"].items():
            metrics[f"train_{key}"] = value
        try:
            batch_size = self.encoder.read_batch_size(batch)
        except:  # noqa: E722
            warn(
                "Unable to parse batch size from data, defaulting to `None` for logging.",
            )
            batch_size = None
        self.log_dict(metrics, on_step=True, prog_bar=True, batch_size=batch_size)
        # step learning rate schedulers at the end of epochs
        if self.trainer.is_last_batch:
            schedulers = self.lr_schedulers()
            if schedulers is not None:
                if not isinstance(schedulers, list):
                    schedulers = [schedulers]
                for s in schedulers:
                    # for schedulers that need a metric
                    if isinstance(s, lr_scheduler.ReduceLROnPlateau):
                        s.step(loss, self.current_epoch)
                    else:
                        s.step(epoch=self.current_epoch)
        if self.hparams.log_embeddings and "embeddings" in batch:
            self._log_embedding(batch["embeddings"])
        return loss_dict

    def _make_normalizers(self) -> dict[str, Normalizer]:
        """
        Applies force specific logic, where we check for the case
        when an energy normalizer is provided by not force. To make
        sure things are consistent, this will automatically add the
        force normalizer with zero mean, and copies the std value from
        the energy normalizer.

        Returns
        -------
        Dict[str, Normalizer]
            Normalizers for each target
        """
        normalizers = super()._make_normalizers()
        if "energy" in normalizers and "force" not in normalizers:
            energy_std = normalizers["energy"].std
            if isinstance(energy_std, torch.Tensor):
                std = energy_std.clone()
                mean = torch.zeros_like(std)
            else:
                # assume it's a float otherwise
                std = energy_std
                mean = 0.0
            force_norm = Normalizer(mean=mean, std=std, device=self.device)
            normalizers["force"] = force_norm
            logger.warning(
                "Energy normalization was specified, but not force. I'm adding it for you."
            )
        # in the case where the energy and force std values are mismatched,
        # we override the force one with the energy one
        if "energy" in normalizers and "force" in normalizers:
            energy_std = normalizers["energy"].std
            force_std = normalizers["force"].std
            if energy_std != force_std:
                normalizers["force"].std = energy_std
                logger.warning(
                    "Force normalization std is overridden with the energy std."
                )
        return normalizers

    @staticmethod
    def get_symmetric_displacement(
        positions: torch.Tensor,
        unit_offsets: torch.Tensor,
        cell: Optional[torch.Tensor],
        src_nodes: torch.Tensor,
        num_graphs: int,
        batch: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """virial/stress computation inspired from:
        https://github.com/ACEsuit/mace/blob/main/mace/modules/utils.py
        https://github.com/mir-group/nequip
        """
        if cell is None:
            cell = torch.zeros(
                num_graphs * 3,
                3,
                dtype=positions.dtype,
                device=positions.device,
            )
        displacement = torch.zeros(
            (num_graphs, 3, 3),
            dtype=positions.dtype,
            device=positions.device,
        )
        displacement.requires_grad_(True)
        symmetric_displacement = 0.5 * (displacement + displacement.transpose(-1, -2))
        positions = positions + torch.einsum(
            "be,bec->bc", positions, symmetric_displacement[batch]
        )
        cell = cell.view(-1, 3, 3)
        cell = cell + torch.matmul(cell, symmetric_displacement)
        shifts = torch.einsum(
            "be,bec->bc",
            unit_offsets,
            cell[batch[src_nodes]],
        )
        return positions, shifts, displacement, symmetric_displacement


@registry.register_task("GradFreeForceRegressionTask")
class GradFreeForceRegressionTask(ScalarRegressionTask):
    def __init__(
        self,
        encoder: nn.Module | None = None,
        encoder_class: type[nn.Module] | None = None,
        encoder_kwargs: dict[str, Any] | None = None,
        loss_func: (
            type[nn.Module] | nn.Module | dict[str, nn.Module | type[nn.Module]] | None
        ) = nn.MSELoss,
        output_kwargs: dict[str, Any] = {},
        **kwargs: Any,
    ) -> None:
        if "task_keys" in kwargs:
            warn(
                f"GradFreeForceRegressionTask does not `task_keys`; "
                f"ignoring passed keys: {kwargs['task_keys']}",
            )
            del kwargs["task_keys"]
        super().__init__(
            encoder,
            encoder_class,
            encoder_kwargs,
            loss_func,
            ["force"],
            output_kwargs,
            **kwargs,
        )

    def _make_output_heads(self) -> nn.ModuleDict:
        modules = {
            "force": OutputHead(3, **self.output_kwargs).to(
                self.device, dtype=self.dtype
            )
        }
        return nn.ModuleDict(modules)

    def _get_targets(
        self,
        batch: dict[str, torch.Tensor | dgl.DGLGraph | dict[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
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
        if "force" not in batch["targets"]:
            raise KeyError(
                f"Force key missing in batch targets: keys found: {batch['targets'].keys()}",
            )
        target_dict = {"force": batch["targets"]["force"]}
        return target_dict

    def forward(
        self,
        batch: dict[str, torch.Tensor | dgl.DGLGraph | dict[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        if "embeddings" in batch:
            embedding = batch.get("embeddings")
        else:
            embedding = self.encoder(batch)
        # check for frame averaging
        if "graph" in batch:
            graph = batch["graph"]
            if hasattr(graph, "ndata"):
                fa_rot = getattr(graph.ndata, "fa_rot", None)
            else:
                fa_rot = getattr(graph, "fa_rot", None)
        outputs = self.process_embedding(embedding, fa_rot)
        return outputs

    def process_embedding(
        self,
        embeddings: Embeddings,
        fa_rot: None | torch.Tensor = None,
    ) -> dict[str, torch.Tensor]:
        """
        Given point/node-level embeddings, predict forces of each point.

        Parameters
        ----------
        embeddings : Embeddings
            Data structure containing system/graph and point/node-level embeddings.

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary containing a ``force`` key that maps to predicted forces
            per point/node
        """
        results = {}
        force_head = self.output_heads["force"]
        forces = force_head(embeddings.point_embedding)
        if isinstance(fa_rot, torch.Tensor):
            natoms = forces.size(0)
            all_forces = []
            # loop over each frame prediction, and transform to guarantee
            # equivariance of frame averaging method
            for frame_idx, frame_rot in fa_rot:
                repeat_rot = torch.repeat_interleave(
                    frame_rot,
                    natoms,
                    dim=0,
                ).to(self.device)
                rotated_forces = (
                    forces[:, frame_idx, :]
                    .view(-1, 1, 3)
                    .bmm(
                        repeat_rot.transpose(1, 2),
                    )
                )
                all_forces.append(rotated_forces.view(natoms, 3))
            # combine all the force data into a single tensor
            forces = torch.stack(all_forces, dim=1)
        # make sure forces are in the right shape
        forces = reduce(forces, "n ... d -> n d", self.embedding_reduction_type, d=3)
        results["force"] = forces
        return results


@registry.register_task("CrystalSymmetryClassificationTask")
class CrystalSymmetryClassificationTask(BaseTaskModule):
    __task__ = "symmetry"

    def __init__(
        self,
        encoder: nn.Module | None = None,
        encoder_class: type[nn.Module] | None = None,
        encoder_kwargs: dict[str, Any] | None = None,
        loss_func: (
            type[nn.Module] | nn.Module | dict[str, nn.Module | type[nn.Module]] | None
        ) = nn.CrossEntropyLoss,
        output_kwargs: dict[str, Any] = {},
        normalize_kwargs: dict[str, float] | None = None,
        freeze_embedding: bool = False,
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
            normalize_kwargs=normalize_kwargs,
            **kwargs,
        )
        self.freeze_embedding = freeze_embedding
        if self.freeze_embedding:
            self.encoder.atom_embedding.requires_grad_(False)

    def _make_output_heads(self) -> nn.ModuleDict:
        # this task only utilizes one output head; 230 possible space groups
        modules = {
            "spacegroup": OutputHead(230, **self.output_kwargs).to(
                self.device, dtype=self.dtype
            )
        }
        return nn.ModuleDict(modules)

    def on_train_batch_start(self, batch: Any, batch_idx: int) -> int | None:
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
            self.task_loss_scaling = self._task_loss_scaling

        return status

    def on_validation_batch_start(
        self,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        self.on_train_batch_start(batch, batch_idx)

    def _get_targets(
        self,
        batch: dict[str, torch.Tensor | dgl.DGLGraph | dict[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        target_dict = {}
        subdict = batch.get("symmetry", None)
        if subdict is None:
            raise ValueError(
                "'symmetry' key is missing from batch, which is needed for space group classification.",
            )
        labels: torch.Tensor = subdict.get("number", None)
        if labels is None:
            raise ValueError(
                "Point group numbers missing from symmetry key, which is needed for symmetry classification.",
            )
        # subtract one for zero-indexing
        labels = labels.long() - 1
        # cast to long type, and make sure it is 1D for cross entropy loss
        if labels.ndim > 1:
            labels = labels.flatten()
        target_dict["spacegroup"] = labels
        return target_dict


@registry.register_task("MultiTaskLitModule")
class MultiTaskLitModule(pl.LightningModule):
    def __init__(
        self,
        *tasks: tuple[str, BaseTaskModule],
        task_scaling: Iterable[float] | None = None,
        task_keys: dict[str, list[str]] | None = None,
        log_embeddings: bool = False,
        log_embeddings_every_n_steps: int = 50,
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
        assert len(tasks) > 0, "No tasks provided."
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
            task_map[dset_name][task.__class__.__name__] = task
            # add dataset names to determine forward logic
            dset_names.add(dset_name)
            # save hyperparameters from subtasks
            subtask_hparams[f"{dset_name}_{task.__class__.__name__}"] = task.hparams
        self.save_hyperparameters(
            {
                "subtask_hparams": subtask_hparams,
                "task_scaling": task_scaling,
                "encoder_opt_kwargs": encoder_opt_kwargs,
                "log_embeddings": log_embeddings,
                "log_embeddings_every_n_steps": log_embeddings_every_n_steps,
            },
        )
        self.task_map = task_map
        self.dataset_names = dset_names
        self.task_scaling = task_scaling
        self.encoder_opt_kwargs = encoder_opt_kwargs
        if task_keys is not None:
            for index, (dataset_name, task_dict) in enumerate(task_keys.items()):
                for relevant_keys in task_dict.values():
                    self._initialize_subtask_output(
                        dataset_name,
                        dict(self.dataset_task_pairs)[dataset_name],
                        task_keys=relevant_keys,
                    )

        self.configure_optimizers()
        self.automatic_optimization = False

    @property
    def task_list(self) -> list[BaseTaskModule]:
        # return a flat list of tasks to iterate over
        modules = []
        for task_group in self.task_map.values():
            for subtask in task_group.values():
                modules.append(subtask)
        return modules

    @property
    def dataset_task_pairs(self) -> list[tuple[str, str]]:
        # Return a list of 2-tuples corresponding to (dataset name, task type)
        pairs = []
        for dataset in self.dataset_names:
            task_types = self.task_map[dataset].keys()
            for task_type in task_types:
                pairs.append((dataset, task_type))
        return pairs

    def configure_optimizers(self) -> list[Optimizer]:
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
                    if isinstance(optimizer, tuple):
                        # unpack the two things if a tuple is returned
                        optimizer, scheduler = optimizer
                    if isinstance(optimizer, list):
                        # we only work with one optimizer
                        optimizer = optimizer[0]
                    # remove all the optimizer parameters, and re-add only the output heads
                    optimizer.param_groups.clear()
                    optimizer.add_param_group({"params": output_head.parameters()})
                    # add optimizer to the pile
                    optimizers.append(optimizer)
                    self.optimizer_names.append((data_key, task_type))
                    index += 1
        assert (
            len(self.optimizer_names) > 1
        ), "Only one optimizer was found for multi-task training."
        if ("Global", "Encoder") not in self.optimizer_names:
            opt_kwargs = {"lr": 1e-4}
            opt_kwargs.update(self.encoder_opt_kwargs)
            optimizers.append(AdamW(self.encoder.parameters(), **opt_kwargs))
            self.optimizer_names.append(("Global", "Encoder"))
        return optimizers

    @property
    def dataset_names(self) -> list[str]:
        return self._dataset_names

    @dataset_names.setter
    def dataset_names(self, values: set | list[str]) -> None:
        if isinstance(values, set):
            values = list(values)
        self._dataset_names = values

    @property
    def task_scaling(self) -> list[float]:
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
    def task_scaling(self, values: Iterable[float] | None) -> None:
        if values is None:
            values = [1.0 for _ in range(self.num_tasks)]
        assert (
            len(values) == self.num_tasks
        ), "Number of provided task scaling values not equal to number of tasks."
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
    def input_grad_keys(self) -> dict[str, list[str]]:
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
        keys = {dset_name: sorted(subkeys) for dset_name, subkeys in keys.items()}
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
        batch: dict[
            str,
            dict[str, torch.Tensor | dgl.DGLGraph | dict[str, torch.Tensor]],
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
            # we determine if it's multidata based on the incoming batch
            # as it should have dataset in its key
            if any(["Dataset" in key for key in batch.keys()]):
                # if this is a multidataset task, loop over each dataset
                # and enable gradients for the inputs that need them
                for dset_name, data in batch.items():
                    input_keys = need_grad_keys.get(dset_name)
                    for key in input_keys:
                        # set require grad for both point cloud and graph tensors
                        if "graph" in data:
                            g = data.get("graph")
                            if isinstance(g, dgl.DGLGraph):
                                if key in g.ndata:
                                    data["graph"].ndata[key].requires_grad_(True)
                            else:
                                # assume it's a PyG graph
                                if key in g:
                                    getattr(g, key).requires_grad_(True)
                        if key in data:
                            target = data.get(key)
                            # for tensors just set them directly
                            if isinstance(target, torch.Tensor):
                                target.requires_grad_(True)
                            else:
                                # assume the remaining case are lists of tensors
                                try:
                                    [t.requires_grad_(True) for t in target]
                                except AttributeError:
                                    pass
            else:
                # in the single dataset case, we just need to loop over a single
                # set of tasks
                input_keys = list(self.input_grad_keys.values()).pop(0)
                for key in input_keys:
                    # set require grad for both point cloud and graph tensors
                    if "graph" in batch:
                        g = batch.get("graph")
                        if isinstance(g, dgl.DGLGraph):
                            if key in g.ndata:
                                batch["graph"].ndata[key].requires_grad_(True)
                        else:
                            # assume it's a PyG graph
                            if key in g:
                                getattr(g, key).requires_grad_(True)
                    if key in batch:
                        target = batch.get(key)
                        # for tensors just set them directly
                        if isinstance(target, torch.Tensor):
                            target.requires_grad_(True)
                        else:
                            # assume the remaining case are lists of tensors
                            try:
                                [t.requires_grad_(True) for t in target]
                            except AttributeError:
                                pass

    def forward(
        self,
        batch: dict[
            str,
            dict[str, torch.Tensor | dgl.DGLGraph | dict[str, torch.Tensor]],
        ],
    ) -> dict[str, dict[str, torch.Tensor]]:
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
            self,
            "needs_dynamic_grads",
            False,
        )  # default to not needing grads
        with dynamic_gradients_context(_grads, self.has_rnn):
            # this function switches of `requires_grad_` for input tensors that need them
            self._toggle_input_grads(batch)
            # compute embeddings for each dataset
            if self.is_multidata:
                for key, data in batch.items():
                    data["embeddings"] = self.encoder(data)
                embeddings = data["embeddings"]
            else:
                batch["embeddings"] = self.encoder(batch)
                embeddings = batch["embeddings"]
            # for single dataset usage, we assume the nested structure isn't used
            if self.is_multidata:
                for key, data in batch.items():
                    subtasks = self.task_map[key]
                    if key not in results:
                        results[key] = {}
                    # finally call the task with the data
                    for task_type, subtask in subtasks.items():
                        results[key][task_type] = subtask.process_embedding(embeddings)
            else:
                # in the single dataset case, we can skip the outer loop
                # and just pass the batch into the subtask
                tasks = list(self.task_map.values()).pop(0)
                for task_type, subtask in tasks.items():
                    results[task_type] = subtask.process_embedding(embeddings)
            return results

    def predict(self, batch: BatchDict) -> dict[str, dict[str, torch.Tensor]]:
        """
        Similar logic to the `BaseTaskModule.predict` method, but implemented
        for the multitask setting.

        The workflow is a linear combination of the two: we run the joint
        embedder once, and then subsequently rely on the `predict` method
        for each subtask to get outputs at their expected scales.

        This method also behaves a little differently from the other multitask
        operations, as it runs a set of data through every single output head,
        ignoring the nominal dataset/subtask unique mapping.

        Parameters
        ----------
        batch
            Input data dictionary, which should correspond to a formatted
            ase.Atoms sample.

        Returns
        -------
        dict[str, dict[str, torch.Tensor]]
            Nested results dictionary, following a dataset/subtask structure.
            For example, {'IS2REDataset': {'ForceRegressionTask': ..., 'ScalarRegressionTask': ...}}
        """
        results = {}
        _grads = getattr(
            self,
            "needs_dynamic_grads",
            False,
        )  # default to not needing grads
        with dynamic_gradients_context(_grads, self.has_rnn):
            # this function switches of `requires_grad_` for input tensors that need them
            self._toggle_input_grads(batch)
            batch["embedding"] = self.encoder(batch)
            # now loop through every dataset/output head pair
            for dset_name, subtask_name in self.dataset_task_pairs:
                subtask = self.task_map[dset_name][subtask_name]
                # use the predict method to get rescaled outputs
                output = subtask.predict(batch)
                # now add it to the rest of the results
                if dset_name not in results:
                    results[dset_name] = {}
                results[dset_name][subtask_name] = output
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
        batch: dict[str, torch.Tensor | dgl.DGLGraph | dict[str, torch.Tensor]],
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
        batch: None
        | (
            dict[
                str,
                dict[str, torch.Tensor | dgl.DGLGraph | dict[str, torch.Tensor]],
            ]
        ) = None,
        task_keys: list[str] | None = None,
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
                f"Unable to initialize output heads for {dataset}-{task_type}; neither batch nor task keys provided.",
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
                    {"params": task_instance.output_heads.parameters()},
                )

    def embed(self, *args, **kwargs) -> Any:
        return self.encoder(*args, **kwargs)

    def _calculate_batch_size(
        self,
        batch: dict[
            str,
            dict[str, torch.Tensor | dgl.DGLGraph | dict[str, torch.Tensor]],
        ],
    ) -> dict[str, int | dict[str, int]]:
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
            elif len(batch["targets"]) > 0:
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
        batch: dict[
            str,
            dict[str, torch.Tensor | dgl.DGLGraph | dict[str, torch.Tensor]],
        ],
        batch_idx: int,
    ) -> dict[str, dict[str, torch.Tensor]]:
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
                        subtask_loss["loss"] * scaling,
                        retain_graph=not is_last_opt,
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
                    loss["loss"] * scaling,
                    retain_graph=not is_last_opt,
                )
                self.on_after_backward()
                loss_logging.update(loss["log"])
        # run before step hooks
        for opt_idx, opt in enumerate(optimizers):
            self.on_before_optimizer_step(opt)
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
        # optionally log embeddings
        if self.hparams.log_embeddings and "embeddings" in batch:
            self._log_embedding(batch["embeddings"])
        return losses

    def validation_step(
        self,
        batch: dict[
            str,
            dict[str, torch.Tensor | dgl.DGLGraph | dict[str, torch.Tensor]],
        ],
        batch_idx: int,
    ) -> dict[str, dict[str, torch.Tensor]]:
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
        # optionally log embeddings
        if self.hparams.log_embeddings and "embeddings" in batch:
            self._log_embedding(batch["embeddings"])
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
            "MultiTask should be reloaded using the `matsciml.models.multitask_from_checkpoint` function instead.",
        )

    def on_save_checkpoint(self, checkpoint):
        """Override checkpoint saving to facilitate EMA weights loading."""
        super().on_save_checkpoint(checkpoint)
        # this case checks for EMA weights, where we give up on
        # the non-averaged weights in favor of the averaged ones
        checkpoint["state_dict"] = self.state_dict()
        if hasattr(self, "ema_module"):
            logger.info("Replacing model weights with exponential moving average.")
            for key, value in self.ema_module.state_dict().items():
                if key != "n_averaged":
                    # remove the module prefix for EMA state
                    tree = key.split(".")
                    renamed_key = ".".join(tree[1:])
                    assert (
                        renamed_key in checkpoint["state_dict"]
                    ), f"Unexpected key in EMA module: {renamed_key}"
                    checkpoint["state_dict"][renamed_key] = value
            # now remove the redundant EMA weights to unblock checkpoint loading later
            for key in list(checkpoint["state_dict"].keys()):
                if "ema_module" in key:
                    del checkpoint["state_dict"][key]

    @classmethod
    def from_pretrained_encoder(cls, task_ckpt_path: str | Path, **kwargs):
        """
        Attempts to instantiate a new task, adopting a previously trained encoder model.

        This function will load in a saved PyTorch Lightning checkpoint,
        copy over the hyperparameters needed to reconstruct the encoder,
        and simply maps the encoder ``state_dict`` to the new instance.

        ``Kwargs`` are passed directly into the creation of the task, and so can
        be thought of as just a task through the typical interface normally.

        Parameters
        ----------
        task_ckpt_path : Union[str, Path]
            Path to an existing task checkpoint file. Typically, this
            would be a PyTorch Lightning checkpoint.

        Examples
        --------
        1. Create a new task simply from training another one

        >>> new_task = ScalarRegressionTask.from_pretrained_encoder(
            "epoch=10-step=100.ckpt"
            )

        2. Create a new task, modifying output heads

        >>> new_taks = ForceRegressionTask.from_pretrained_encoder(
            "epoch=5-step=12516.ckpt",
            output_kwargs={
                "num_hidden": 3,
                "activation": "nn.ReLU"
            }
        )
        """
        if isinstance(task_ckpt_path, str):
            task_ckpt_path = Path(task_ckpt_path)
        assert (
            task_ckpt_path.exists()
        ), "Encoder checkpoint filepath specified but does not exist."
        ckpt = torch.load(task_ckpt_path)
        for key in ["encoder_class", "encoder_kwargs"]:
            assert (
                key in ckpt["hyper_parameters"]
            ), f"{key} expected to be in hyperparameters, but was not found."
            # copy over the data for the new task
            kwargs[key] = ckpt["hyper_parameters"][key]
        # construct the new task with random weights
        task = cls(**kwargs)
        # this only copies over encoder weights, and removes the 'encoder.'
        # pattern from keys
        encoder_weights = {
            key.replace("encoder.", ""): tensor
            for key, tensor in ckpt["state_dict"].items()
            if "encoder." in key
        }
        # load in pre-trained weights
        task.encoder.load_state_dict(encoder_weights)
        return task

    def _log_embedding(self, embeddings: Embeddings) -> None:
        """
        This maps the appropriate logging function depending on what
        logger was used, and saves the graph and node level embeddings.

        Some services like ``wandb`` are able to do some nifty embedding
        analyses online using these embeddings.

        Parameters
        ----------
        embeddings : Embeddings
            Data structure containing embeddings from the encoder.
        """
        log_freq = self.hparams.log_embeddings_every_n_steps
        global_step = self.trainer.global_step
        # only log embeddings at the same cadence as everything else
        if self.logger is not None and (global_step % log_freq) == 0:
            exp = self.logger.experiment
            sys_z = embeddings.system_embedding.detach().cpu()
            node_z = embeddings.point_embedding.detach().cpu()
            if isinstance(self.logger, pl_loggers.WandbLogger):
                # this import is okay here since we need it for the logger anyway
                import wandb

                cols = [f"D{i}" for i in range(sys_z.size(-1))]
                exp.log(
                    {"graph_embeddings": wandb.Table(columns=cols, data=sys_z.tolist())}
                )
                if isinstance(embeddings.point_embedding, torch.Tensor):
                    # TODO: should add labels to the nodes based on graph index
                    exp.log(
                        {
                            "node_embeddings": wandb.Table(
                                columns=cols, data=node_z.tolist()
                            )
                        }
                    )
            elif isinstance(self.logger, pl_loggers.TensorBoardLogger):
                exp.add_embedding(
                    sys_z,
                    tag=f"graph_embeddings_{self.trainer.global_step}",
                )
                if isinstance(embeddings.point_embedding, torch.Tensor):
                    # TODO: should add labels to the nodes based on graph index
                    exp.add_embedding(
                        node_z,
                        tag=f"node_embeddings_{self.trainer.global_step}",
                    )
            else:
                pass


@registry.register_task("OpenCatalystInference")
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
            f"{self.__class__.__name__} is solely used for OpenCatalyst leaderboard submissions; please call 'predict' from trainer.",
        )

    def training_step(self, *args: Any, **kwargs: Any) -> None:
        self._raise_inference_error()

    def validation_step(self, *args: Any, **kwargs: Any) -> None:
        self._raise_inference_error()

    def test_step(self, *args: Any, **kwargs: Any) -> None:
        self._raise_inference_error()

    @abstractmethod
    def predict_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> Any: ...


@registry.register_task("IS2REInference")
class IS2REInference(OpenCatalystInference):
    def __init__(
        self,
        pretrained_model: AbstractEnergyModel | ScalarRegressionTask,
    ) -> None:
        assert isinstance(
            pretrained_model,
            (AbstractEnergyModel, ScalarRegressionTask),
        ), "IS2REInference expects a pretrained energy model or 'ScalarRegressionTask' as input."
        super().__init__(pretrained_model)

    def forward(self, batch: BatchDict) -> DataDict:
        predictions = self.model(batch)
        return predictions


@registry.register_task("S2EFInference")
class S2EFInference(OpenCatalystInference):
    def __init__(self, pretrained_model: ForceRegressionTask) -> None:
        assert isinstance(
            pretrained_model,
            ForceRegressionTask,
        ), "S2EFInference expects a pretrained 'ForceRegressionTask' instance as input."
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
            chunk_split = torch.split(fixed, natoms)
            chunk_ids = []
            for chunk in chunk_split:
                ids = (len(chunk) - sum(chunk)).cpu().numpy().astype(int)
                chunk_ids.append(int(ids))

            predictions["chunk_ids"] = chunk_ids
        return predictions

    def on_predict_batch_end(
        self,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        # reset gradients to ensure no contamination between batches
        self.zero_grad(set_to_none=True)


@registry.register_task("NodeDenoisingTask")
class NodeDenoisingTask(BaseTaskModule):
    __task__ = "pretraining"
    """
    This implements a node position denoising task, as described by Zaidi _et al._,
    ICLR 2023.

    This task is paired with the `NoisyPositions` pretraining data transform,
    which generates the noise. A single output head is used to predict the noise
    for every atom, using the MSE between the predicted and actual noise as the
    loss function.
    """

    def __init__(
        self,
        encoder: nn.Module | None = None,
        encoder_class: type[nn.Module] | None = None,
        encoder_kwargs: dict[str, Any] | None = None,
        loss_func: type[nn.Module] | nn.Module | None = nn.MSELoss,
        task_keys: list[str] | None = None,
        output_kwargs: dict[str, Any] = {},
        lr: float = 0.0001,
        weight_decay: float = 0,
        embedding_reduction_type: str = "mean",
        normalize_kwargs: dict[str, float] | None = None,
        scheduler_kwargs: dict[str, dict[str, Any]] | None = None,
        **kwargs,
    ) -> None:
        if task_keys is not None:
            logger.warning(
                "Task keys were passed to NodeDenoisingTask, but is not used."
            )
        task_keys = ["denoise"]
        super().__init__(
            encoder,
            encoder_class,
            encoder_kwargs,
            loss_func,
            task_keys,
            output_kwargs,
            lr,
            weight_decay,
            embedding_reduction_type,
            normalize_kwargs,
            scheduler_kwargs,
            **kwargs,
        )

    def _make_output_heads(self) -> nn.ModuleDict:
        # make a single output head for noise prediction applied to nodes
        denoise = OutputHead(3, **self.output_kwargs).to(self.device, dtype=self.dtype)
        return nn.ModuleDict({"denoise": denoise})

    def _filter_task_keys(
        self,
        keys: list[str],
        batch: dict[str, torch.Tensor | dgl.DGLGraph | dict[str, torch.Tensor]],
    ) -> list[str]:
        """
        For the denoising task, we will only ever target the "denoise" key.

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
        return ["denoise"]

    def process_embedding(self, embeddings: Embeddings) -> dict[str, torch.Tensor]:
        """
        Override the base process embedding method, since we are assumed to only
        have a single output head and we need to use the point/node-level embeddings.

        Parameters
        ----------
        embeddings : Embeddings
            Embeddings data structure containing graph and node-level embeddings.

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary with a single 'denoise' key, corresponding to the
            predicted noise.
        """
        head = self.output_heads["denoise"]
        # prediction node noise
        pred_noise = head(embeddings.point_embedding)
        return {"denoise": pred_noise}

    def forward(
        self,
        batch: BatchDict,
    ) -> dict[str, torch.Tensor]:
        """
        Modified forward call for denoising positions.

        The goal of this task is to predict noise, given noisy coordinates,
        and for this to happen we substitute the noise-free positions temporarily
        for the noisy ones to prevent interference with other tasks.

        Parameters
        ----------
        batch : BatchDict
            Batch of data samples

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary output from ``process_embedding``

        Raises
        ------
        KeyError:
            Raises a ``KeyError`` ff the noisy positions are not found
            in either the graph or point cloud dictionary.
        """
        if "graph" in batch:
            graph = batch["graph"]
            if hasattr(graph, "ndata"):
                target = graph.ndata
            else:
                target = graph
        else:
            target = batch
        if "noisy_pos" not in target:
            raise KeyError(
                "'noisy_pos' was not found in data structure, please add the"
                " NoisyPositions pretraining transform, and/or check that"
                " 'noisy_pos' is included in the graph transform ``node_keys``."
            )
        temp_pos = target["pos"].clone().detach()
        # swap out positions for the noisy ones
        target["pos"] = target["noisy_pos"]
        if "embeddings" in batch:
            embedding = batch.get("embeddings")
        else:
            embedding = self.encoder(batch)
        outputs = self.process_embedding(embedding)
        target["pos"] = temp_pos
        return outputs
