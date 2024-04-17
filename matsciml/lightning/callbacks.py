from __future__ import annotations

import gc
import json
import os
from collections.abc import Sequence
from datetime import datetime
from logging import DEBUG, getLogger
from pathlib import Path
from time import time
from typing import Any, Callable, Dict, Iterator, Optional

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import BasePredictionWriter, Callback
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from torch import distributed as dist
from torch import nn
from torch.optim import Optimizer

from matsciml.common.packages import package_registry
from matsciml.datasets.utils import concatenate_keys
from matsciml.models.base import BaseTaskModule


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
        algorithm: str = "norm",
        verbose: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.value = value
        self.algorithm = algorithm
        self.kwargs = kwargs
        self.verbose = verbose

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
        opt_idx: int,
    ) -> None:
        for parameter in pl_module.parameters():
            self.algorithm(parameter, self.value, **self.kwargs)
            if self.verbose:
                print(parameter)


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

    def on_before_optimizer_step(
        self,
        trainer: Trainer,
        task: BaseTaskModule,
        optimizer: Optimizer,
    ) -> None:
        with torch.no_grad():
            org_weights = self._first_step(optimizer)
        with torch.enable_grad():
            loss = task._compute_losses(self.batch)
            loss = self._get_loss(loss)
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
