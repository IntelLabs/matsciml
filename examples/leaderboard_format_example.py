from __future__ import annotations

import pytorch_lightning as pl

from matsciml.lightning.callbacks import LeaderboardWriter
from matsciml.lightning.data_utils import IS2REDGLDataModule, is2re_devset
from matsciml.models import GraphConvModel, IS2RELitModule

# import the callback responsible for aggregating and formatting
# prediction results

"""
This example walks through the bare minimum example to generate a leaderboard
ready submission file using the IS2RE development set.

Here, we use a freshly initialized `GraphConvModel`, but you
can replace this with a `load_from_checkpoint` to grab pretrained weights.

To fit with the PyTorch Lightning abstraction, we rely on the `predict` pipeline/loop
to perform inference. The code below will run through 5 batches, then save the
result to `inference_results/IS2RELitModule/GraphConvModel/{datetime}.npz`, in
a format that should be evalAI ready.
"""

model = IS2RELitModule(GraphConvModel(100, 1), lr=1e-3, gamma=0.9)
dm = IS2REDGLDataModule(predict_path=is2re_devset, batch_size=8)

# limit to 5 prediction batches to demonstrate pipeline
trainer = pl.Trainer(
    limit_predict_batches=5,
    callbacks=[LeaderboardWriter("inference_results")],
)
# run the prediction loop
trainer.predict(model, datamodule=dm)
