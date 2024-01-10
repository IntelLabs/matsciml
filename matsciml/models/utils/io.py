from __future__ import annotations

from pathlib import Path
from typing import Union

import torch

from matsciml.models import base
from matsciml.models.base import MultiTaskLitModule


def multitask_from_checkpoint(ckpt_path: str | Path) -> MultiTaskLitModule:
    """
    Utility function to load a MultiTaskLitModule from a checkpoint.

    This is implemented as a separate function as opposed to the `load_from_checkpoint`
    due to some nuances with how `@classmethod` scopes affect class instantiation.
    This function basically loads in the checkpoint file, recreates the tasks,
    then loads in the state dict.

    Parameters
    ----------
    ckpt_path : Union[str, Path]
        Path to a PyTorch Lightning checkpoint file for a MultiTaskLitModule

    Returns
    -------
    MultiTaskLitModule
        Reloaded MultiTaskLitModule
    """
    if isinstance(ckpt_path, str):
        ckpt_path = Path(ckpt_path)
    assert ckpt_path.exists(), f"Checkpoint file not found; passed {ckpt_path}"
    ckpt_data = torch.load(ckpt_path)
    hparams = ckpt_data["hyper_parameters"]
    tasks = []
    for key, subdict in hparams["subtask_hparams"].items():
        # unpack dict, then grab task class
        dset_name, task_name = key.split("_")
        task_class = getattr(base, task_name)
        tasks.append((dset_name, task_class(**subdict)))
    creation_kwargs = {}
    for key in ["task_scaling", "task_keys"]:
        creation_kwargs[key] = hparams.get(key, None)
    # try and see if there are additional encoder kwargs to be passed
    creation_kwargs.update(hparams.get("encoder_opt_kwargs", {}))
    # create the multitask module from tasks
    task_module = MultiTaskLitModule(*tasks, **creation_kwargs)
    # load weights into model
    task_module.load_state_dict(ckpt_data["state_dict"])
    return task_module
