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
    "ForceRegressionTask": ["energy", "forces"],
    "GradFreeForceRegressionTask": ["forces"],
}


class AbstractStrategy(ABC):
    @abstractmethod
    def merge_outputs(self, *args, **kwargs) -> dict[str, float | np.ndarray]: ...

    def parse_outputs(
        self, output_dict: DataDict, task: MultiTaskLitModule, *args, **kwargs
    ) -> dict[str, dict[str, float | torch.Tensor]]:
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
        # loop over the task map
        for dset_name in task.task_map.keys():
            for subtask_name, subtask in task.task_map[dset_name].items():
                sub_results = {}
                pos_fields = __task_property_mapping__.get(subtask, None)
                if pos_fields is None:
                    continue
                else:
                    for key in pos_fields:
                        output = output_dict[dset_name][subtask_name].detach()
                        if key == "energy":
                            output = output.item()
                        sub_results[key] = output
                if len(sub_results) == 0:
                    raise RuntimeError(
                        f"Expected {subtask_name} to have {pos_fields} but got nothing."
                    )
                results[dset_name] = sub_results
        return results


class AverageTasks(AbstractStrategy):
    def merge_outputs(self): ...
