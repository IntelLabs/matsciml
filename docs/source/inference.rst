Inference
=========

"Inference" can be a bit of an overloaded term, and this page is broken down into different possible
downstream use cases for trained models.

Parity plots and model evaluations
----------------------------------

The simplest/most straightforward thing to check the performance of a model is to look beyond reduced metrics; i.e. anything that
has been averaged over batches, epochs, etc. Parity plots help verify linear relationships between predictions and ground truths
by simply iterating over the evaluation subset of data, averaging.

The ``ParityInferenceTask`` helps perform this task by using the PyTorch Lightning ``predict`` pipelines. With a pre-trained
``matsciml`` task checkpoint, you simply need to run the following:

.. codeblock:: python

    import pytorch_lightning as pl

    from matsciml.models.inference import ParityInferenceTask
    from matsciml.lightning import MatSciMLDataModule

    # configure data module the way that you need to
    dm = MatSciMLDataModule(
        dataset="NameofDataset",
        pred_split="/path/to/lmdb/split",
        batch_size=64   # this is just to amoritize model calls
    )
    task = ParityInferenceTask.from_pretrained_checkpoint("/path/to/checkpoint")

    trainer = pl.Trainer()   # optionally, configure logger/limit_predict_batches
    trainer.predict(task, datamodule=dm)


The default ``Trainer`` settings will create a ``lightning_logs`` directory, followed by an experiment
number. Within it, once your inference run completes, there will be a ``inference_data.json`` that you
can then load in. The data is sorted by the name of the target (e.g. ``energy``, ``bandgap``), under
these keys, ``predictions`` and ``targets``.

.. note::

    For developers, this is handled by the ``matsciml.models.inference.ParityData`` class. This is
    mainly to standardize the output and provide a means to serialize the data as JSON.



.. autoclass:: matsciml.models.inference.ParityInferenceTask
   :members:



Performing molecular dynamics simulations
-----------------------------------------

Currently, the main method of interfacing with dynamical simulations is through the ``ase`` package.
Documentation for this is ongoing, but examples can be found under ``examples/interfaces``.

.. autoclass:: matsciml.interfaces.ase.MatSciMLCalculator
    :members:
