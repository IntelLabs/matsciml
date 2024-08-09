from __future__ import annotations

import gc
import json
import os
from collections.abc import Sequence
from datetime import datetime
from logging import DEBUG, getLogger
from pathlib import Path
from time import time
from copy import copy
from typing import Any, Callable, Dict, Iterator, Literal, Optional
from queue import Queue

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import BasePredictionWriter, Callback
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from pytorch_lightning import loggers as pl_loggers
from torch import distributed as dist
from torch import nn
from torch.optim import Optimizer
from dgl import DGLGraph
from scipy.signal import correlate
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn, update_bn

from matsciml.common.packages import package_registry
from matsciml.datasets.utils import concatenate_keys
from matsciml.models.base import BaseTaskModule
from matsciml.common.types import Embeddings, BatchDict
from matsciml.lightning.loss_scaling import BaseScalingSchedule

__all__ = [
    "LeaderboardWriter",
    "GradientCheckCallback",
    "UnusedParametersCallback",
    "ThroughputCallback",
    "ForwardNaNDetection",
    "ManualGradientClip",
    "MonitorGradients",
    "GarbageCallback",
    "InferenceWriter",
    "CodeCarbonCallback",
    "SAM",
    "TrainingHelperCallback",
    "ModelAutocorrelation",
    "ExponentialMovingAverageCallback",
    "LossScalingScheduler",
]


class LeaderboardWriter(BasePredictionWriter):
    """
    Callback/logger utility that is used in conjunction with the `predict`
    pipeline to generate formatted results ready to be submitted to
    the evalAI leaderboard.

    The way this is setup is slightly clunky: we perform the gather step
    at the end of the epoch (which occurs before the `LightningModule.on_predict_epoch_end`
    for some reason), and on the head rank will save the result to a directory
    of your choosing, nested as {task}/{model_name}/{datetime}.npz
    """

    def __init__(self, output_path: str | Path) -> None:
        super().__init__(write_interval="epoch")
        self.output_path = output_path

    @property
    def output_path(self) -> Path:
        return self._output_path

    @output_path.setter
    def output_path(self, value: str | Path) -> None:
        if isinstance(value, str):
            value = Path(value)
        os.makedirs(value, exist_ok=True)
        self._output_path = value

    @property
    def now(self) -> str:
        now = datetime.now()
        return now.strftime("%Y%m%d_%H%M%S")

    def write_on_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        predictions: Sequence[dict[str, torch.Tensor]],
        batch_indices: Sequence[Any] | None,
    ) -> None:
        task_name = pl_module.__class__.__name__
        # TODO refactor this to use model attr instead of gnn
        gnn_name = pl_module.gnn.__class__.__name__
        # for all workers, gather up the inference results
        world_predictions = pl_module.all_gather(predictions)[0]
        # not the best way to do this, but we need this method to
        # run on all workers for the sync to happen
        if rank_zero_only.rank == 0:
            keys = world_predictions[0].keys()
            results = {key: [] for key in keys}
            for prediction in world_predictions:
                for key, value in prediction.items():
                    if any([isinstance(v, str) for v in value]):
                        results[key].extend(value)
                    else:
                        if key == "chunk_ids":
                            value = [value.cpu() for value in value]
                            results[key].extend(value)
                        else:
                            results[key].extend(value.cpu())

            for key, values in results.items():
                if key == "ids":
                    pass
                else:
                    results[key] = torch.stack(values).numpy()

            target = self.output_path.joinpath(f"{task_name}/{gnn_name}/{self.now}.npz")
            # make the directory in case it doesn't exist
            target.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(target, **results)
            print(f"\nSaved NPZ log file to: {target}\n")


def deep_tensor_trawling(input_data: tuple[dict[str, Any]]):
    if len(input_data) == 1:
        input_data = input_data[0]
    results = {}
    for key, value in input_data.items():
        if isinstance(value, dict):
            results[key] = deep_tensor_trawling(value)
        elif isinstance(value, torch.Tensor):
            results[key] = value.detach()
        elif key == "graph":
            for subkey, node_data in value.ndata.items():
                results[subkey] = node_data.detach()
    return results


def forward_nan_hook(module: nn.Module, inputs: Any, output: torch.Tensor) -> None:
    """
    Create a hook that will save the input/output tensors to a module if there are NaNs
    detected in the output tensor.

    To use this function, register it as a forward hook using `nn.Module.register_forward_hook`.

    Parameters
    ----------
    module : nn.Module
        PyTorch nn.Module/layer of interest
    input : torch.Tensor
        Input tensor fed into the nn.Module
    output : torch.Tensor
        Output tensor as a result of module(input)
    """
    if torch.any(output.isnan()):
        setattr(
            module,
            "nan_detection",
            {"input": deep_tensor_trawling(inputs), "output": output.detach()},
        )


class GradientCheckCallback(Callback):
    """
    Callback to monitor gradients in a model. Just before the optimizer is
    stepped, we will inspect the gradients for every single learnable parameter.
    If there are NaNs in the gradients, we will print out the parameter and step
    number, and then zero out the gradients. Otherwise, we will inspect the
    gradient norm and ensure it's above a specified threshold.
    """

    def __init__(
        self,
        thres: float = 1e-2,
        num_steps: int = -1,
        verbose: bool = False,
    ) -> None:
        super().__init__()
        self.thres = thres
        self.logger = getLogger("pytorch_lightning")
        if verbose:
            self.logger.setLevel(DEBUG)
        self.num_steps = num_steps

    def on_before_optimizer_step(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        optimizer: Optimizer,
    ) -> None:
        step_number = trainer.global_step
        # this checks to make sure we're still running the nan check
        if self.num_steps <= step_number:
            gradients = []
            for name, param in pl_module.named_parameters():
                if param.requires_grad and param.grad is not None:
                    # check if there are NaNs as well
                    if torch.any(torch.isnan(param.grad)):
                        grad_fn = getattr(param, "grad_fn", None)
                        if grad_fn:
                            node_name = f"/{grad_fn.name()}"
                        else:
                            node_name = ""
                        self.logger.debug(
                            f"Step number {step_number} has NaN gradients for parameter {name}{node_name}. Zeroing!",
                        )
                        # zero out gradients
                        param.grad.zero_()
                    else:
                        grad_norm = param.detach().norm()
                        # detach from the computational graph and just check the norm value
                        if grad_norm < self.thres:
                            gradients.append((name, grad_norm.item()))
            if len(gradients) > 0:
                msg = (
                    f"Parameters with gradient norm less than {self.thres}: {gradients}"
                )
                self.logger.debug(msg)


class UnusedParametersCallback(Callback):
    def on_before_optimizer_step(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        optimizer: Optimizer,
        opt_idx: int,
    ) -> None:
        for name, parameter in pl_module.named_parameters():
            if parameter.grad is None:
                print(
                    f"{name} has no gradients and is not part of the computational graph.",
                )


class ThroughputCallback(Callback):
    def __init__(self, log_dir: str, batch_size: int) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.log_dir = log_dir
        self.record = []

    @property
    def workers(self) -> int:
        if dist.is_initialized():
            return dist.get_world_size()
        return 1

    @property
    def log_dir(self) -> Path:
        return self._log_dir

    @log_dir.setter
    def log_dir(self, path: str | Path) -> None:
        if isinstance(path, str):
            path = Path(path)
        if not path.exists():
            os.makedirs(path, exist_ok=True)
        self._log_dir = path

    @property
    def start_time(self) -> int:
        return self._start_time

    @start_time.setter
    def start_time(self, value: int) -> None:
        self._start_time = value

    @property
    def end_time(self) -> int:
        return self._end_time

    @end_time.setter
    def end_time(self, value: int) -> None:
        self._end_time = value

    @property
    def elapsed(self) -> int:
        return self._end_time - self._start_time

    @property
    def throughput(self) -> float:
        return (self.batch_size * self.workers) / self.elapsed

    @rank_zero_only
    def on_train_batch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch: Any,
        batch_idx: int,
    ) -> None:
        # get time with nanosecond precision
        self.start_time = time()

    @rank_zero_only
    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch: Any,
        batch_idx: int,
    ) -> None:
        self.end_time = time()
        self.counter = batch_idx + 1
        self.record.append(self.formatted_result)

    @property
    def formatted_result(self) -> dict[str, int | float]:
        result = {
            "elapsed": self.elapsed,
            "throughput": self.throughput,
            "counter": self.counter,
            "batches": self.counter * self.batch_size * self.workers,
        }
        return result

    @rank_zero_only
    def on_fit_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        epoch = trainer.current_epoch
        target = self.log_dir.joinpath(f"epoch{epoch}_throughput_measurement.json")
        with open(target, "w+") as write_file:
            json.dump(self.record, write_file)


class ForwardNaNDetection(Callback):
    def __init__(self, output_path: str | Path) -> None:
        super().__init__()
        if isinstance(output_path, str):
            output_path = Path(output_path)
        self.output_path = output_path
        os.makedirs(self.output_path, exist_ok=True)

    @property
    def target_file(self) -> Path:
        return self.output_path.joinpath(f"forward_nan_check_step{self.step_num}.log")

    def on_fit_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        for child in pl_module.children():
            child.register_forward_hook(forward_nan_hook)

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch: Any,
        batch_idx: int,
    ) -> None:
        self.step_num = trainer.global_step
        all_data = []
        for name, child in pl_module.named_children():
            nan_data = getattr(child, "nan_detection", None)
            if nan_data is not None:
                nan_data["name"] = name
                # cast tenors to strings for saving
                nan_data["input"] = str(nan_data["input"])
                nan_data["output"] = str(nan_data["output"])
                all_data.append(nan_data)
        if len(all_data) != 0:
            with open(self.target_file, "w+") as write_file:
                for entry in all_data:
                    write_file.write(
                        " ".join([f"{key}: {value}" for key, value in entry.items()]),
                    )
                write_file.write("\n")


class ManualGradientClip(Callback):
    def __init__(
        self,
        value: float,
        algorithm: Literal["norm", "value"] = "norm",
        **kwargs,
    ) -> None:
        super().__init__()
        self.value = value
        self.algorithm = algorithm
        self.kwargs = kwargs

    @property
    def algorithm(self) -> Callable:
        return self._algorithm

    @algorithm.setter
    def algorithm(self, value: str) -> None:
        if value == "norm":
            method = nn.utils.clip_grad_norm_
        else:
            method = nn.utils.clip_grad_value_
        self._algorithm = method

    def on_before_optimizer_step(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        optimizer: Optimizer,
    ) -> None:
        for parameter in pl_module.parameters():
            self.algorithm(parameter, self.value, **self.kwargs)


class MonitorGradients(Callback):
    """
    This callback is useful for pulling out gradients for analysis.

    The verbose argument will print to standard output a chunk of gradients
    live, and perform an `allclose` on current and previous gradients to
    make sure each batch provides gradient signals that are not the same.
    """

    def __init__(
        self,
        step_frequency: int,
        verbose: bool = False,
        eps: float = 1e-10,
    ) -> None:
        super().__init__()
        self.step_frequency = step_frequency
        self.verbose = verbose
        self.eps = eps

    def on_before_optimizer_step(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        optimizer: Optimizer,
        opt_idx: int,
    ) -> None:
        # figure out what the step number is from optimizer
        state = optimizer.state[optimizer.param_groups[0]["params"][-1]]
        if len(state) == 0:
            step = 0
        else:
            step = int(state["step"])
        if step % self.step_frequency == 0:
            tensors = []
            for parameter in optimizer.state.keys():
                if parameter.grad is not None:
                    tensors.append(parameter.grad.flatten())
                else:
                    pass
            joint_state = torch.concat(tensors)
            torch.save(joint_state, f"step{step}_opt{opt_idx}_grads.pt")

    def on_after_backward(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        encoder = pl_module.encoder
        tensors = []
        no_grads = []
        for name, parameter in encoder.named_parameters():
            if parameter.grad is not None:
                tensors.append(parameter.grad.flatten())
            elif parameter.grad is None or parameter.grad.sum() == 0.0:
                no_grads.append(name)
        joint_state = torch.concat(tensors)
        if hasattr(self, "last_state"):
            is_close = torch.allclose(joint_state, self.last_state)
        else:
            is_close = False
        self.last_state = joint_state
        if self.verbose:
            print(
                f"Step: {trainer.global_step} - Grads: {joint_state[:50]} - Equal? {is_close} - Zero grads: {no_grads}\n",
            )


class GarbageCallback(Callback):
    def __init__(self, frequency: int) -> None:
        super().__init__()
        self.frequency = frequency

    def on_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """
        Run garbage collection at the end of a batch.

        Frequency of which garbage collection occurs is set by class argument
        and on the `trainer.global_step` attribute.

        Parameters
        ----------
        trainer : pl.Trainer
            Instance of PyTorch Lightning trainer
        pl_module : pl.LightningModule
            Instance of PyTorch Lightning module
        """
        step = trainer.global_step
        if step % self.frequency == 0:
            _ = gc.collect()


class InferenceWriter(BasePredictionWriter):
    def __init__(self, output_dir: str | Path) -> None:
        """
        Set up the ``InferenceWriter`` callback.

        This relies on the ``predict`` loop in PyTorch Lightning to generate
        prediction, and writes out the results in ``torch`` format for
        each worker (i.e. can use DDP for inference). Results will be aggregated
        once the dataset has been exhausted.

        Parameters
        ----------
        output_dir : Union[str, Path]
            Path to a folder to dump results to.

        Examples
        --------
        Add the writer as a callback to ``Trainer``

        >>> import pytorch_lightning as pl
        >>> from matsciml.lightning.callbacks import InferenceWriter
        >>> trainer = pl.Trainer(callbacks=[InferenceWriter("./predictions")])
        >>> trainer.predict(...)
        """
        super().__init__(write_interval="epoch")
        self.output_dir = output_dir

    @property
    def output_dir(self) -> Path:
        return self._output_dir

    @output_dir.setter
    def output_dir(self, value: str | Path) -> None:
        if isinstance(value, str):
            value = Path(value)
        os.makedirs(value, exist_ok=True)
        self._output_dir = value

    def write_on_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        predictions: Sequence[Any],
        batch_indices: Sequence[Any] | None,
    ) -> None:
        predictions = concatenate_keys(predictions[0])
        # downcast to float16 to save some space
        for key, value in predictions.items():
            if isinstance(value, torch.FloatTensor):
                predictions[key] = value.to(torch.float16)
        rank = trainer.global_rank
        path = self.output_dir.joinpath(f"results_rank{rank}.pt")
        torch.save(predictions, path)


if package_registry["codecarbon"]:
    from codecarbon import EmissionsTracker, OfflineEmissionsTracker

    class CodeCarbonCallback(Callback):
        """
        Integrates `codecarbon` functionality to the end-to-end workflows
        such as "fit", "test", and "predict". Currently, we just segment into
        these different tasks, and the measurement is integrated throughout
        until we are done with training, etc.

        In the future this could be further broken down into measuring power
        consumption at specific parts of workflows.
        """

        def __init__(
            self,
            output_file: str | Path = "emissions.pt",
            output_dir: str | Path | None = None,
            offline: bool = True,
            **kwargs,
        ):
            """
            Instantiate a ``CodeCarbonCallback`` object.

            This configures the matsciml workflow to use ``codecarbon``
            and estimate emissions footprint for training, testing, and
            inference tasks.

            Parameters
            ----------
            output_file : str | Path, default "emissions.pt"
                Filename to save the emissions data to. The default
                value will save it to an "emissions.pt" file.
            output_dir : str | Path | None, default None
                Directory to house the output file. The default behavior
                is set to ``None``, which will try and share the ``log_dir``
                of a logger to consolidate results.
            offline : bool
                Flag to use the offline workflow, which is the default case.

            Raises
            ------
            KeyError
                If using offline mode and no country ISO code is provided,
                ``codecarbon`` will refuse to work.
            """
            super().__init__()
            track = OfflineEmissionsTracker if offline else EmissionsTracker
            # override some settings to make it compatible with intended usage
            kwargs["save_to_file"] = False
            kwargs["save_to_api"] = False
            kwargs.setdefault("project_name", "matsciml-experiment")
            # remove redundant config keys
            for key in ["output_file", "output_dir"]:
                if key in kwargs:
                    del kwargs[key]
            if offline and "country_iso_code" not in kwargs:
                raise KeyError(
                    "Offline mode specified but no country ISO code provided."
                )
            self.tracker = track(**kwargs)
            self.output_file = output_file
            self._temp_output_dir = output_dir
            self.reset_data()

        def reset_data(self) -> None:
            self.data = {key: [] for key in ["fit", "test", "predict"]}

        def dump_data(self, trainer: pl.Trainer):
            """
            Writes the data to disk via ``torch.save``.

            This will try and be a little clever by borrowing a logger's
            output directory if it exists, and fall back to defaults
            otherwise.

            Parameters
            ----------
            trainer
                Instance of ``pl.Trainer``, which is used to access the
                logger instances as well.
            """
            log_dir = self._temp_output_dir
            # if nothing was specified
            if not log_dir and trainer.logger:
                # try and get it from the logger
                # if we have multiple loggers, find the first
                # `log_dir` to use
                if isinstance(trainer.logger, list):
                    for logger in trainer.logger:
                        log_dir = getattr(logger, "log_dir", None)
                        if log_dir:
                            break
                # when we only have a single logger just grab it
                else:
                    log_dir = getattr(trainer.logger, "log_dir", None)
            # fallback to default value otherwise
            if not log_dir:
                log_dir = "./emissions_data"
            log_dir = Path(log_dir)
            # this case exists if we aren't using logging and need to create
            # a folder
            if not log_dir.exists():
                log_dir.mkdir(parents=True)
            output = log_dir.joinpath(self.output_file)
            torch.save(self.data, str(output.absolute()))

        def teardown(
            self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: str
        ) -> None:
            super().teardown(trainer, pl_module, stage)
            # trigger a dump if and only if we are actively monitoring
            if self.tracker._scheduler and self.tracker._active_task:
                taskname = self.tracker._active_task
                emissions_data = self.tracker.stop_task(taskname)
                self.tracker.stop()
                self.data[taskname].append(emissions_data)
                self.dump_data(trainer)

        def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
            self.tracker.start()
            self.tracker.start_task("fit")

        def on_predict_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
            self.tracker.start()
            self.tracker.start_task("predict")

        def on_test_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
            self.tracker.start()
            self.tracker.start_task("test")

        def on_exception(
            self,
            trainer: "pl.Trainer",
            pl_module: "pl.LightningModule",
            exception: BaseException,
        ) -> None:
            # trigger this case when we break or fail out of training and whatnot
            # unexpectedly
            if not (stage := self.tracker._active_task):
                stage = ""
            self.teardown(trainer, pl_module, stage)
            super().on_exception(trainer, pl_module, exception)


class SAM(Callback):
    def __init__(self, rho: float = 0.05, adaptive: bool = False) -> None:
        """
        Set up the ``SAM (Sharpness Aware Minimization)`` callback.
        https://arxiv.org/abs/2010.01412

        This implementation is adapted from https://github.com/davda54/sam.

        SAM (Sharpness Aware Minimization) simultaneously minimizes loss
        value and loss sharpness it seeks parameters that lie in neighborhoods
        having uniformly low loss improving model generalization.

        The training will run twice as slow because SAM needs two forward-backward
        passes to estimate the "sharpness-aware" gradient.

        If you're using gradient clipping, make sure to change only the magnitude
        of gradients, not their direction.

        Parameters
        ----------
        rho : float
            A hyperparameter determining the scale of regularization for
            sharpness-aware minimization. Defaults to 0.05.
        adaptive : bool
            A boolean flag indicating whether to adaptively normalize weights.
            Defaults to False.

        Examples
        --------

        >>> import pytorch_lightning as pl
        >>> from matsciml.lightning.callbacks import SAM
        >>> trainer = pl.Trainer(callbacks=[SAM()])
        >>> trainer.fit(...)
        """

        super().__init__()
        self.rho = rho
        self.adaptive = adaptive

    @staticmethod
    def _get_params(optimizer: Optimizer) -> Iterator[torch.Tensor]:
        for group in optimizer.param_groups:
            for param in group["params"]:
                if not isinstance(param, torch.Tensor):
                    raise TypeError(f"expected Tensor, but got: {type(param)}")
                yield param

    @staticmethod
    def _get_loss(step_output: Any) -> Optional[torch.Tensor]:
        if step_output is None:
            return None
        if isinstance(step_output, torch.Tensor):
            return step_output
        return step_output.get("loss")

    def on_train_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: Any,
        batch_idx: int,
    ) -> None:
        self.batch = batch
        self.batch_idx = batch_idx

    def extract_optimizer_specific_loss(self, task, optimizer, loss):
        optimizer_names = copy(task.optimizer_names)
        opt_idx = [opt.optimizer == optimizer for opt in task.optimizers()].index(True)
        loss_keys = optimizer_names[opt_idx]
        if loss_keys == ("Global", "Encoder"):
            optimizer_names.pop(opt_idx)
            global_loss = 0
            for dataset, task in optimizer_names:
                if loss.get(dataset, None) is not None:
                    global_loss += loss[dataset][task]["loss"]
            return {"loss": global_loss}
        # When some datasets have less samples than others, they wont have a loss value
        if loss_keys[0] not in loss:
            loss = {"loss": None}
        else:
            for key in loss_keys:
                loss = loss[key]
        return loss

    def is_optimizer_used(self, task, optimizer):
        # Check if only one optimizer is used (single task)
        if isinstance(task.optimizers(), Optimizer):
            return True
        # Otherwise, see if the specific optimizer we are looking at is used in the current batch.
        # If it is not present, this means there will be no loss value and all of the parameters
        # gradients will be None.
        optimizer_names = copy(task.optimizer_names)
        opt_idx = [opt.optimizer == optimizer for opt in task.optimizers()].index(True)
        used_optimizer_names = self.batch.keys()
        if optimizer_names[opt_idx][0] in list(used_optimizer_names):
            return True
        else:
            return False

    def on_before_optimizer_step(
        self,
        trainer: Trainer,
        task: BaseTaskModule,
        optimizer: Optimizer,
    ) -> None:
        optimizer_is_used = self.is_optimizer_used(task, optimizer)
        if optimizer_is_used:
            with torch.no_grad():
                org_weights = self._first_step(optimizer)
            with torch.enable_grad():
                loss = task._compute_losses(self.batch)
                # this is for the multitask case where there is more than on optimizer
                if not isinstance(task.optimizers(), Optimizer):
                    loss = self.extract_optimizer_specific_loss(task, optimizer, loss)
                loss = self._get_loss(loss)
                if loss is not None:
                    if torch.isfinite(loss):
                        trainer.strategy.backward(loss, optimizer=optimizer)
            with torch.no_grad():
                self._second_step(optimizer, org_weights)

    def _norm_weights(self, p: torch.Tensor) -> torch.Tensor:
        return torch.abs(p) if self.adaptive else torch.ones_like(p)

    def _grad_norm(self, optimizer: Optimizer) -> torch.Tensor:
        param_norms = torch.stack(
            [
                (self._norm_weights(p) * p.grad).norm()
                for p in self._get_params(optimizer)
                if isinstance(p.grad, torch.Tensor)
            ]
        )
        return param_norms.norm()

    def _first_step(self, optimizer: Optimizer) -> Dict[torch.Tensor, torch.Tensor]:
        """
        org_weights dictionary stores original weights and perturbed weights
        """
        scale = self.rho / (self._grad_norm(optimizer) + 1e-5)
        org_weights: Dict[torch.Tensor, torch.Tensor] = {}
        for p in self._get_params(optimizer):
            if p.grad is None:
                continue
            org_weights[p] = p.detach().clone()
            e_w = (torch.pow(p, 2) if self.adaptive else 1.0) * p.grad * scale
            p.add_(e_w)
        optimizer.zero_grad()
        return org_weights

    def _second_step(
        self, optimizer: Optimizer, org_weights: Dict[torch.Tensor, torch.Tensor]
    ) -> None:
        for p in self._get_params(optimizer):
            if p.grad is None:
                continue
            p.data = org_weights[p]


def embedding_magnitude_hook(
    module: nn.Module, input: BatchDict, output: Embeddings
) -> None:
    """
    Forward hook that will inspect an embedding output.

    This checks for two properties of graph-level and node-level embeddings:
    the magnitude of the median tells us if the values are a lot larger than
    what we might typically expect, and the variance tells us if the embeddings
    are effectively collapsing.

    Parameters
    ----------
    module : nn.Module
        Nominally a PyTorch module, but we actually expect an encoder.
    input : BatchDict
        Batch of samples to process
    output : Embeddings
        Expected to be an embedding data structure. If not, we don't
        fail the run, but posts a critical message.
    """
    logger = getLogger("matsciml.helper")
    logger.setLevel("INFO")
    if isinstance(output, Embeddings):
        # check the magnitude of both node and system level embeddings
        if output.system_embedding is not None:
            sys_z = output.system_embedding.detach().cpu()
            # calculate representative statistics
            sys_z_med = sys_z.median().abs().item()
            sys_z_var = sys_z.var().item()
            if sys_z_med > 10.0:
                logger.warning(
                    f"Median system/graph embedding value is greater than 10 ({sys_z_med})"
                )
            if sys_z_var <= 1e-5:
                logger.warning(
                    f"Variance in system/graph embedding is quite small ({sys_z_var})"
                )
        if output.point_embedding is not None:
            node_z = output.point_embedding.detach().cpu()
            # calculate representative statistics
            node_z_med = node_z.median().abs().item()
            node_z_var = node_z.var().item()
            if node_z_med > 10.0:
                logger.warning(
                    f"Median node embedding value is greater than 10 ({node_z_med})"
                )
            if node_z_var <= 1e-5:
                logger.warning(
                    f"Variance in node embedding is quite small ({node_z_var})"
                )
    else:
        logger.critical(
            f"Hooked module does not produce an embedding data structure! {module}"
        )


class TrainingHelperCallback(Callback):
    def __init__(
        self,
        small_grad_thres: float = 1e-3,
        update_freq: int = 50,
        encoder_hook: bool = True,
        record_param_norm_history: bool = True,
    ) -> None:
        """
        Initializes a ``TrainingHelperCallback``.

        The purpose of this callback is to provide some typical
        heuristics that are useful for diagnosing how training
        is progressing. The behavior of this callback is twofold:
        (1) emit warning messages to the user, indicating that
        there are irregularities like missing gradients, and low
        variance in embeddings; (2) send some of these observations
        to loggers like ``TensorBoardLogger`` and ``WandbLogger``
        for asynchronous viewing.

        Parameters
        ----------
        small_grad_thres : float, default 1e-3
            Threshold for detecting when gradients for particular
            parameters are considered small. This helps identify
            layers that could benefit with some residual connections.
        update_freq : int, default 50
            Frequency of which to run checks with this callback.
            This can be increased to make messages less spammy.
        encoder_hook : bool, default True
            If True, we register a forward hook with the model's
            encoder that is specifically designed for ``matsciml``
            usage. This hook will inspect graph and node level
            embeddings, particularly variance in dimensions, to
            identify feature collapse.
        record_param_norm_history : bool, default True
            If True, will log tensor norms to ``tensorboard`` or
            ``wandb`` services.
        """
        super().__init__()
        self.logger = getLogger("matsciml.helper")
        self.logger.setLevel("INFO")
        self.small_grad_thres = small_grad_thres
        self.update_freq = update_freq
        self.encoder_hook = encoder_hook
        self.record_param_norm_history = record_param_norm_history

    def on_fit_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        """
        This attaches the embedding hook, which inspects the embeddings
        to make sure there is sufficient variance, or if the values are
        too big.
        """
        if self.encoder_hook:
            pl_module.encoder.register_forward_hook(embedding_magnitude_hook)
            self.logger.info("Registered embedding monitor")

    def on_train_epoch_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        """Sets an internal batch index tracker for activity."""
        self.batch_idx = 0

    def on_train_batch_start(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        batch: Any,
        batch_idx: int,
    ) -> None:
        """
        Triggering at the beginning of a training batch, this is where all
        the checks pertaining to input data should be made. For now,
        we check whether or not the coordinates are bounded between 0,1
        which may indicate that the coordinates are fractional which may
        not be intended.
        """
        self.batch_idx = batch_idx
        if self.is_active:
            # look at atom positions for irregularities
            if "graph" in batch:
                g = batch["graph"]
                if isinstance(g, DGLGraph):
                    pos = g.ndata["pos"]
                else:
                    pos = g.pos
            else:
                # we assume there are positions, otherwise there are bigger
                # problems than running this check
                pos = batch["pos"]
            min_pos, max_pos = pos.min().item(), pos.max().item()
            if min_pos >= 0.0 and max_pos <= 1.0:
                self.logger.warning(
                    "Coordinates are small and might be fractional, which may not be intended."
                )

    @property
    def is_active(self) -> bool:
        """Determines whether or not to perform an update."""
        return (self.batch_idx % self.update_freq) == 0

    @staticmethod
    def encoder_head_comparison(
        pl_module: pl.LightningModule,
        log_history,
        python_logger,
        global_step: int | None = None,
    ):
        """
        Make a comparison of weight norms in the encoder and output head stack.

        The heuristic being checked here is if the encoder weights are a lot smaller
        than the output head, the encoder may end up being ignored entirely and
        the output heads are just overfitting to the data. This check doesn't prove
        that is happening, but provides an indication of it.

        Parameters
        ----------
        pl_module
            Nominally a generic ``LightningModule``, but we expect the
            model to have an encoder and an output head module dict.
        log_history : bool
            Default True, whether to log the weight norm values to an
            experiment tracker.
        python_logger : Logger
            Logger for the Python side to raise the warning message.
        """
        # compare encoder and output head weights
        encoder_norm_vals = []
        output_norm_vals = []
        for parameter in pl_module.encoder.parameters():
            encoder_norm_vals.append(parameter.detach().norm().cpu().item())
        for head in pl_module.output_heads.values():
            for parameter in head.parameters():
                output_norm_vals.append(parameter.detach().norm().cpu().item())
        encoder_norm_vals = np.array(encoder_norm_vals)
        output_norm_vals = np.array(output_norm_vals)
        encoder_median = np.median(encoder_norm_vals)
        output_median = np.median(output_norm_vals)
        if encoder_median < (2.0 * output_median):
            python_logger.warning(
                "Median encoder weights are significantly smaller than output heads:"
                " encoder median norm: {encoder_median:.3e},"
                " output head: {output_median:.3e}"
            )
        # optionally record to a supported service as well
        # this nominally should work for multiple loggers
        if log_history and len(pl_module.loggers) > 0:
            for pl_logger in pl_module.loggers:
                log_service = pl_logger.experiment
                encoder_norm_vals = torch.from_numpy(encoder_norm_vals).float()
                output_norm_vals = torch.from_numpy(output_norm_vals).float()
                if isinstance(log_service, pl_loggers.TensorBoardLogger):
                    log_service.add_histogram(
                        "encoder_weight_norm", encoder_norm_vals, global_step
                    )
                    log_service.add_histogram(
                        "outputhead_weight_norm", output_norm_vals, global_step
                    )
                elif isinstance(log_service, pl_loggers.WandbLogger):
                    log_service.log({"encoder_weight_norm": encoder_norm_vals})
                    log_service.log({"outputhead_weight_norm": output_norm_vals})

    def on_before_optimizer_step(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        optimizer: Optimizer,
    ) -> None:
        """
        This stage checks for problems pertaining to parameter weights
        and gradients, triggering before the optimizer is stepped.
        We check to make sure the gradient norm is reasonably sized,
        as well as making sure that the output head weigts don't get
        significantly larger than the encoder.
        """
        if self.is_active:
            log_service = pl_module.logger
            # loop through parameter related checks
            grad_norm_vals = []
            for name, parameter in pl_module.named_parameters():
                if parameter.requires_grad:
                    if parameter.grad is None:
                        self.logger.warning(
                            f"Parameter {name} has no gradients, but should!"
                        )
                    else:
                        grad_norm = parameter.grad.norm()
                        if grad_norm.abs() < self.small_grad_thres:
                            self.logger.warning(
                                f"Parameter {name} has small gradient norm - {grad_norm}"
                            )
                        grad_norm_vals.append(grad_norm.detach().cpu().item())
            # track gradient norm for the whole model
            grad_norm_vals = torch.FloatTensor(grad_norm_vals)
            if isinstance(log_service, pl_loggers.TensorBoardLogger):
                log_service.experiment.add_histogram(
                    "gradient_norms", grad_norm_vals, global_step=trainer.global_step
                )
            elif isinstance(log_service, pl_loggers.WandbLogger):
                log_service.experiment.log(
                    {"gradient_norms": torch.FloatTensor(grad_norm_vals)}
                )
            self.encoder_head_comparison(
                pl_module,
                self.record_param_norm_history,
                self.logger,
                trainer.global_step,
            )


class ModelAutocorrelation(Callback):
    def __init__(
        self,
        buffer_size: int = 100,
        sampled: bool = True,
        sample_frac: float = 0.05,
        analyze_grads: bool = True,
        analyze_every_n_steps: int = 50,
    ) -> None:
        """
        Initializes a ``ModelAutocorrelation`` callback.

        The purpose of this callback is to track parameters and optionally
        gradients over time, and periodically calculate the autocorrelation
        spectrum to see how correlated parameters and gradients are throughout
        the training process.

        Parameters
        ----------
        buffer_size : int, default 100
            Number of steps worth of parameters/gradients to keep in
            the correlation window. If the buffer is too small, the
            autocorrelation might not be particularly meaningful; if
            it's too big, it may impact training throughput.
        sampled : bool, default True
            If True, we ``sample_frac`` worth of elements from every
            parameter tensor. The False case has not yet been implemented,
            but is intended to track the whole model.
        sample_frac : float, default 0.05
            Fraction of a given parameter/gradient tensor to track.
            Larger values give a better picture for how the whole
            model is behaving, while fewer samples mean less impact
            but a poorer description.
        analyze_grads : bool, default True
            If True, perform the autocorrelation procedure for gradients
            as well as parameters. This may give a better indication of
            dynamics over parameters alone.
        analyze_every_n_steps : int, default 50
            Frequency to carry out the autocorrelation analysis. Note
            that sampling is done at every training step, regardless
            of this value. Instead, this determines how often we do the
            autocorrelation calculation and logging.

        Raises
        ------
        NotImplementedError
            If ``sampled=False``, which has not yet been implemented.
        """
        super().__init__()
        self.buffer_size = buffer_size
        if not sampled:
            raise NotImplementedError(
                "Only sampled analysis mode is currently supported."
            )
        self.sampled = sampled
        self.sample_frac = sample_frac
        self.analyze_grads = analyze_grads
        self.analyze_every_n_steps = analyze_every_n_steps

    @staticmethod
    def sample_parameters(
        model: nn.Module, indices: dict[str, torch.Tensor], collect_grads: bool
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """
        Collect elements from parameter and gradient tensors of the
        target model based on a dictionary of indices.

        Indices are expected to run over the number elements flattened
        tensors (i.e. ``Tensor.numel``).

        Parameters
        ----------
        model : nn.Module
            PyTorch model to track
        indices : dict[str, torch.Tensor]
            Dictionary mapping for layer name and corresponding
            parameter tensor
        collect_grads
            If True, gradients will also be recorded. Evidently
            this means twice the storage requirement.

        Returns
        -------
        tuple[np.ndarray, np.ndarray | None]
            If ``collect_grads`` is True, a 2-tuple of arrays
            will be returned, corresponding to the sampled
            parameters and gradients. If False, the latter will
            just be None.
        """
        collected_params = []
        collected_grads = []
        for name, parameter in model.named_parameters():
            idx = indices.get(name, None)
            if idx is not None:
                elements = parameter.flatten()[idx].detach().cpu().numpy()
                collected_params.append(elements)
                if collect_grads and parameter.grad is not None:
                    collected_grads.append(
                        parameter.grad.flatten()[idx].detach().cpu().numpy()
                    )
        if collect_grads and len(collected_grads) > 0:
            return np.hstack(collected_params), np.hstack(collected_grads)
        else:
            return np.hstack(collected_params), None

    def run_analysis(self, logger: pl_loggers.Logger):
        """
        Perform the autocorrelation analysis.

        This function will convert the history buffer into arrays
        and pass them to ``_calculate_autocorrelation``. If we have
        a logger (either ``wandb`` or ``tensorboard``), we will
        log the correlation spectra to these services as well.

        Parameters
        ----------
        logger : pl_loggers.Logger
            Abstract PyTorch Lightning logger instance. While it is
            technically abstract, only ``WandbLogger`` and ``TensorBoardLogger``
            are supported right now
        """
        param_history = np.vstack(self.history["params"].queue)
        param_corr = self._calculate_autocorrelation(param_history)
        # now log the spectrum
        if isinstance(logger, pl_loggers.WandbLogger):
            from wandb.plot import line_series

            logger.experiment.log(
                {
                    "param_autocorrelation": line_series(
                        xs=[i for i in range(param_history.shape[0])],
                        ys=param_corr.tolist(),
                        title="Parameter autocorrelation",
                        xname="Steps",
                    )
                }
            )
        elif isinstance(logger, pl_loggers.TensorBoardLogger):
            logger.experiment.add_image(
                "param_autocorrelation",
                param_corr,
                global_step=self.global_step,
                dataformats="WH",
            )
        else:
            raise NotImplementedError(
                "Only WandbLogger and TensorBoardLogger are currently supported."
            )

        if self.analyze_grads:
            grad_history = np.vstack(self.history["grads"].queue)
            grad_corr = self._calculate_autocorrelation(grad_history)
            if isinstance(logger, pl_loggers.WandbLogger):
                from wandb.plot import line_series

                logger.experiment.log(
                    {
                        "grad_autocorrelation": line_series(
                            xs=[i for i in range(grad_history.shape[0])],
                            ys=grad_corr.tolist(),
                            title="Gradient autocorrelation",
                            xname="Steps",
                        )
                    }
                )
            elif isinstance(logger, pl_loggers.TensorBoardLogger):
                logger.experiment.add_image(
                    "grad_autocorrelation",
                    grad_corr,
                    global_step=self.global_step,
                    dataformats="WH",
                )

    @staticmethod
    def _calculate_autocorrelation(history: np.ndarray) -> np.ndarray:
        """
        Use ``scipy.signal.correlate`` to calculate the autocorrelation
        for parameters and optionally gradients.

        This spectrum tells you the degree of correlation between training
        steps in the recent history for every parameter/gradient element
        being tracked. The rescaling is done unintelligently, and for
        purely aesthetic reasons.

        Parameters
        ----------
        history : np.ndarray
            NumPy 2D array; the first dimension is time step, and the
            second is parameter/gradient element.

        Returns
        -------
        np.ndarray
            NumPy 2D array; the first dimension is time step, and the
            second corresponds to autocorrelation power/signal.
        """
        assert history.ndim == 2, "Expected history to be 2D!"
        # normalizing by variance explodes, so just make it relative
        corr = correlate(history, history, mode="same")
        corr = (corr - corr.min(axis=0)) / (corr.max(axis=0) - corr.min(axis=0))
        return corr

    def on_fit_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        """
        Setup the callback tracking. For sampling mode, we generate random
        indices for every parameter in the model (that isn't lazy) that
        corresponds to parameters/gradients we will consistently track
        throughout training.

        Parameters
        ----------
        trainer : pl.Trainer
            PyTorch Lightning trainer instance
        pl_module : pl.LightningModule
            PyTorch Lightning module to track
        """
        if self.sampled:
            indices = {}
            for name, parameter in pl_module.named_parameters():
                if not isinstance(parameter, torch.nn.UninitializedParameter):
                    numel = parameter.numel()
                    indices[name] = torch.randperm(numel)[
                        : int(numel * self.sample_frac)
                    ]
            self.indices = indices
        # queue structure is used to manage the history with a finite
        # number of elements
        self.history = {
            "params": Queue(self.buffer_size),
            "grads": Queue(self.buffer_size),
        }

    @property
    def global_step(self) -> int:
        return self._global_step

    @global_step.setter
    def global_step(self, value: int) -> None:
        self._global_step = value

    @property
    def is_active(self) -> bool:
        """Used to determine whether the correlation analysis will be carried out."""
        return (
            self.global_step % self.analyze_every_n_steps
        ) == 0 and self.global_step != 0

    def on_train_batch_start(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        batch: Any,
        batch_idx: int,
    ) -> None:
        self.global_step = trainer.global_step

    def on_before_optimizer_step(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", loss: torch.Tensor
    ) -> None:
        """
        Triggers before the optimizer is stepped, adding the parameters and
        optionally gradients to the history.

        If the current step matches the analysis frequency, carry out the
        autocorrelation analysis and log the spectrum.

        Parameters
        ----------
        trainer : pl.Trainer
            PyTorch Lightning trainer instance
        pl_module : pl.LightningModule
            PyTorch Lightning module to track
        loss : torch.Tensor
            Loss value; unused
        """
        params, grads = self.sample_parameters(
            pl_module, self.indices, self.analyze_grads
        )
        self.history["params"].put(params)
        if self.analyze_grads:
            self.history["grads"].put(grads)
        # remove the oldest part of history first if we're full
        if self.history["params"].full():
            _ = self.history["params"].get()
        if self.history["grads"].full():
            _ = self.history["grads"].get()
        if self.is_active:
            self.run_analysis(pl_module.logger)


class ExponentialMovingAverageCallback(Callback):
    def __init__(
        self,
        decay: float = 0.99,
        verbose: bool | Literal["WARN", "INFO", "DEBUG"] = "WARN",
    ) -> None:
        """
        Initialize an exponential moving average callback.

        This callback attaches a ``ema_module`` attribute to
        the current training task, which duplicates the model
        weights that are tracked with an exponential moving
        average, parametrized by the ``decay`` value.

        This will double the memory footprint of your model,
        but has been shown to considerably improve generalization.

        Parameters
        ----------
        decay : float
            Exponential decay factor to apply to updates.
        """
        super().__init__()
        self.decay = decay
        self.logger = getLogger("matsciml.ema_callback")
        if isinstance(verbose, bool):
            if not verbose:
                verbose = "WARN"
            else:
                verbose = "INFO"
        if isinstance(verbose, str):
            assert verbose in [
                "WARN",
                "INFO",
                "DEBUG",
            ], "Invalid verbosity setting in EMA callback."
        self.logger.setLevel(verbose)

    def on_fit_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        # check to make sure the module has no lazy layers
        for layer in pl_module.modules():
            if isinstance(layer, nn.modules.lazy.LazyModuleMixin):
                if layer.has_uninitialized_params():
                    raise RuntimeError(
                        "EMA callback does not support lazy layers. Please "
                        "re-run without using lazy layers."
                    )
        # in the case that there is already an EMA state we don't initialize
        if hasattr(pl_module, "ema_module"):
            self.logger.info(
                "Task has an existing EMA state; not initializing a new one."
            )
            self.ema_module = pl_module.ema_module
        else:
            # hook to the task module and in the current callback
            ema_module = AveragedModel(
                pl_module, multi_avg_fn=get_ema_multi_avg_fn(self.decay)
            )
            self.logger.info("Task does not have an existing EMA state; creating one.")
            # setting the callback ema_module attribute allows ease of access
            self.ema_module = ema_module
            pl_module.ema_module = ema_module

    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs,
        batch: Any,
        batch_idx: int,
    ) -> None:
        self.logger.info("Updating EMA state.")
        pl_module.ema_module.update_parameters(pl_module)

    def on_fit_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        loader = trainer.train_dataloader
        self.logger.info("Fit finished - updating EMA batch normalization state.")
        update_bn(loader, pl_module.ema_module)


class LossScalingScheduler(Callback):
    def __init__(
        self,
        *schedules: BaseScalingSchedule,
        log_level: Literal["INFO", "DEBUG", "WARNING", "CRITICAL"] = "INFO",
    ) -> None:
        """
        Callback for dynamically adjusting loss scaling values over
        the course of training, a la curriculum learning.

        This class is configured by supplying a list of schedules
        as args; see `matsciml.lightning.loss_scaling` module for
        available schedules. Each schedule instance has a `key`
        attribute that points it to the corresponding task key
        as set in the Lightning task module (e.g. `energy`, `force`).

        Parameters
        ----------
        args : BaseScalingSchedule
            Scaling schedules for as many tasks as being performed.
        """
        super().__init__()
        assert len(schedules) > 0, "Must pass individual schedules to loss scheduler!"
        self.schedules = schedules
        self._logger = getLogger("matsciml.loss_scaling_scheduler")
        self._logger.setLevel(log_level)
        self._logger.debug(f"Configured {len(self.schedules)} schedules.")
        self._logger.debug(
            f"Schedules have {[s.key for s in self.schedules]} task keys."
        )

    def on_fit_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        trainer.datamodule.setup("fit")
        for schedule in self.schedules:
            # check to make sure the schedule key actually exists in the task
            if schedule.key not in pl_module.task_keys:
                raise KeyError(
                    f"Schedule for {schedule.key} expected, but not specified as a task key!"
                )
            # schedules grab what information they need from the
            # trainer and task modules
            schedule.setup(trainer, pl_module)
            self._logger.debug("Configured {schedule.key} schedule.")

    def _step_schedules(
        self, pl_module: "pl.LightningModule", stage: Literal["step", "epoch"]
    ) -> None:
        """Base function to step schedules according to what stage we are in."""
        for schedule in self.schedules:
            if schedule.step_frequency == stage:
                target_key = schedule.key
                self._logger.debug(
                    f"Attempting to advance {target_key} schedule on {stage}."
                )
                try:
                    new_scaling_value = schedule.step()
                    pl_module.task_loss_scaling[target_key] = new_scaling_value
                    self._logger.debug(
                        f"Advanced {target_key} to new value: {new_scaling_value}"
                    )
                except StopIteration:
                    self._logger.warning(
                        f"{target_key} has run out of scheduled values; this may be unintentional."
                    )

    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        self._step_schedules(pl_module, "step")

    def on_train_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        self._step_schedules(pl_module, "epoch")
