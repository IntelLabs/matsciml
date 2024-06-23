from __future__ import annotations

from abc import abstractmethod, ABC

import torch
import numpy as np

from matsciml.models.base import (
    MultiTaskLitModule,
)
from matsciml.common.types import DataDict


__task_property_mapping__ = {
    "ScalarRegressionTask": ["energy", "dipole"],
    "ForceRegressionTask": ["energy", "force"],
    "GradFreeForceRegressionTask": ["force"],
}


__all__ = ["AverageTasks"]


class AbstractStrategy(ABC):
    @abstractmethod
    def merge_outputs(
        self,
        outputs: dict[str, dict[str, float | torch.Tensor]]
        | dict[str, list[float | torch.Tensor]],
        *args,
        **kwargs,
    ) -> dict[str, float | np.ndarray]: ...

    def parse_outputs(
        self, output_dict: DataDict, task: MultiTaskLitModule, *args, **kwargs
    ) -> tuple[
        dict[str, dict[str, float | torch.Tensor]],
        dict[str, list[float | torch.Tensor]],
    ]:
        """
        Map the task results into their appropriate fields.

        Expected output looks like:
        {"IS2REDataset": {"energy": ..., "forces": ...}, ...}

        Parameters
        ----------
        output_dict : DataDict
            Multitask/multidata output from the ``MultiTaskLitModule``
            forward pass.
        task : MultiTaskLitModule
            Instance of the task module. This allows access to the
            ``task.task_map``, which tells us which dataset/subtask
            is mapped together.

        Returns
        -------
        dict[str, dict[str, float | torch.Tensor]]
            Dictionary mapping of results per dataset. The subdicts
            correspond to the extracted outputs, per subtask (e.g.
            energy/force from the IS2REDataset head).
        dict[str, list[float | torch.Tensor]]
            For convenience, this provides the same data without
            differentiating between datasets, and instead, sorts
            them by the property name (e.g. {"energy": [...]}).

        Raises
        ------
        RuntimeError:
            When no subresults are returned for a dataset that is
            expected to have something on the basis that a task
            _should_ produce something, e.g. ``ForceRegressionTask``
            should yield energy/force, and if it doesn't produce
            anything, something is wrong.
        """
        results = {}
        per_key_results = {}
        # loop over the task map
        for dset_name in task.task_map.keys():
            for subtask_name, subtask in task.task_map[dset_name].items():
                sub_results = {}
                pos_fields = __task_property_mapping__.get(subtask_name, None)
                if pos_fields is None:
                    continue
                else:
                    for key in pos_fields:
                        output = output_dict[dset_name][subtask_name].get(key, None)
                        # this means the task _can_ output the key but was
                        # not included in the actual training task keys
                        if output is None:
                            continue
                        if isinstance(output, torch.Tensor):
                            output = output.detach()
                        if key == "energy":
                            # squeeze is applied just in case we have too many
                            # extra dimensions
                            output = output.squeeze().item()
                        sub_results[key] = output
                        # add to per_key_results as another sorting
                        if key not in per_key_results:
                            per_key_results[key] = []
                        per_key_results[key].append(output)
                if len(sub_results) == 0:
                    raise RuntimeError(
                        f"Expected {subtask_name} to have {pos_fields} but got nothing."
                    )
                results[dset_name] = sub_results
        return results, per_key_results

    @abstractmethod
    def run(self, output_dict: DataDict, task: MultiTaskLitModule, *args, **kwargs): ...

    def __call__(
        self, output_dict: DataDict, task: MultiTaskLitModule, *args, **kwargs
    ) -> dict[str, float | np.ndarray]:
        aggregated_results = self.run(output_dict, task, *args, **kwargs)
        # TODO: homogenize keys so we don't have to do stuff like this :P
        if "force" in aggregated_results:
            aggregated_results["forces"] = aggregated_results["force"]
        return aggregated_results


class AverageTasks(AbstractStrategy):
    def merge_outputs(
        self, outputs: dict[str, list[float | torch.Tensor]], *args, **kwargs
    ) -> dict[str, float | np.ndarray]:
        joined_results = {}
        for key, results in outputs.items():
            if isinstance(results[0], float):
                merged_results = sum(results) / len(results)
            elif isinstance(results[0], torch.Tensor):
                results = torch.stack(results, dim=0)
                merged_results = results.mean(dim=0).numpy()
            else:
                raise TypeError(
                    f"Only floats and tensors are supported for merging; got {type(results[0])} for key {key}."
                )
            joined_results[key] = merged_results
        return joined_results

    def run(
        self, output_dict: DataDict, task: MultiTaskLitModule, *args, **kwargs
    ) -> dict[str, float | np.ndarray]:
        _, per_key_results = self.parse_outputs(output_dict, task)
        aggregated_results = self.merge_outputs(per_key_results)
        return aggregated_results
