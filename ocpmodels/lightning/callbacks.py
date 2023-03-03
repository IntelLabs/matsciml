import os
from typing import Dict, Union, Any
from logging import getLogger
from time import time
from pathlib import Path
import json

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.rank_zero import rank_zero_only
import torch
from torch import nn
from torch import distributed as dist
from torch.optim import Optimizer


def forward_nan_hook(module: nn.Module, input: torch.Tensor, output: torch.Tensor) -> None:
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
        setattr(module, "nan_detection", {"input": input.detach(), "output": output.detach()})


class GradientCheckCallback(Callback):
    """
    Callback to monitor gradients in a model. Just before the optimizer is
    stepped, we will inspect the gradients for every single learnable parameter.
    If there are NaNs in the gradients, we will print out the parameter and step
    number, and then zero out the gradients. Otherwise, we will inspect the
    gradient norm and ensure it's above a specified threshold.
    """

    def __init__(self, thres: float = 1e-2, num_steps: int = -1) -> None:
        super().__init__()
        self.thres = thres
        self.logger = getLogger("pytorch_lightning")
        self.num_steps = num_steps

    def on_before_optimizer_step(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        optimizer: Optimizer,
        opt_idx: int,
    ) -> None:
        step_number = trainer.global_step
        # this checks to make sure we're still running the nan check
        if self.num_steps <= step_number:
            gradients = []
            for (name, param) in pl_module.named_parameters():
                if param.requires_grad and param.grad is not None:
                    # check if there are NaNs as well
                    if torch.any(torch.isnan(param.grad)):
                        self.logger.debug(
                            f"Step number {step_number} has NaN gradients for parameter {name}. Zeroing!"
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
        def log_dir(self, path: Union[str, Path]) -> None:
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
            trainer: "pl.Trainer",
            pl_module: "pl.LightningModule",
            batch: Any,
            batch_idx: int,
        ) -> None:
            # get time with nanosecond precision
            self.start_time = time()

        @rank_zero_only
        def on_train_batch_end(
            self,
            trainer: "pl.Trainer",
            pl_module: "pl.LightningModule",
            outputs,
            batch: Any,
            batch_idx: int,
        ) -> None:
            self.end_time = time()
            self.counter = batch_idx + 1
            self.record.append(self.formatted_result)

        @property
        def formatted_result(self) -> Dict[str, Union[int, float]]:
            result = {
                "elapsed": self.elapsed,
                "throughput": self.throughput,
                "counter": self.counter,
                "batches": self.counter * self.batch_size * self.workers,
            }
            return result

        @rank_zero_only
        def on_fit_end(
            self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
        ) -> None:
            epoch = trainer.current_epoch
            target = self.log_dir.joinpath(f"epoch{epoch}_throughput_measurement.json")
            with open(target, "w+") as write_file:
                json.dump(self.record, write_file)


class ForwardNaNDetection(Callback):
    def __init__(self, output_path: Union[str, Path]) -> None:
        super().__init__()
        if isinstance(output_path, str):
            output_path = Path(output_path)
        self.output_path = output_path
        os.makedirs(self.output_path, exist_ok=True)

    @property
    def target_file(self) -> Path:
        return self.output_path.joinpath(f"forward_nan_check_step{self.step_num}.log")

    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        for child in pl_module.children():
            child.register_forward_hook(forward_nan_hook)

    def on_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
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
                    write_file.write(" ".join([f"{key}: {value}" for key, value in entry.items()]))
                write_file.write("\n")
