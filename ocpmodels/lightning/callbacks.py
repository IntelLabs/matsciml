from typing import Union, Sequence, Optional, Any, Dict
from pathlib import Path
import os
from datetime import datetime

import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from pytorch_lightning.callbacks import BasePredictionWriter


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

    def __init__(self, output_path: Union[str, Path]) -> None:
        super().__init__(write_interval="epoch")
        self.output_path = output_path

    @property
    def output_path(self) -> Path:
        return self._output_path

    @output_path.setter
    def output_path(self, value: Union[str, Path]) -> None:
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
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        predictions: Sequence[Dict[str, torch.Tensor]],
        batch_indices: Optional[Sequence[Any]],
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
            joint_results = {}
            for key in keys:
                combined_result = torch.stack(
                    [batch[key] for batch in world_predictions]
                )
                joint_results[key] = combined_result.flatten(0, -2).numpy()
            target = self.output_path.joinpath(f"{task_name}/{gnn_name}/{self.now}.npz")
            # make the directory in case it doesn't exist
            target.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(target, **joint_results)
