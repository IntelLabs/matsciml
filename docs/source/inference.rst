Inference
=========

"Inference" can be a bit of an overloaded term, and this page is broken down into different possible
downstream use cases for trained models.

Task ``predict`` and ``forward`` methods
----------------------------------------

``matsciml`` tasks implement separate ``forward`` and ``predict`` methods. Both take a
``BatchDict`` as input, and the latter wraps the former. The difference, however, is that
``predict`` is intended for inference use primarily because it will also take care of
reversing the normalization procedure, if they were provided during training, *and* perhaps
more importantly, will ensure that the exponential moving average weights are used instead
of the training ones.

In the special case of force prediction (as a derivative of the energy) tasks, you should
only need to specify normalization ``kwargs`` for energy: the scale value is taking automatically
from the energy value, and applied to forces.

In short, if you are writing functionality that requires unnormalized outputs (e.g. ``ase`` calculators),
please ensure you are using ``predict`` instead of ``forward`` directly.


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
these keys, ``predictions`` and ``targets``. Note that ``pred_split`` does not necessarily have to be
a completely different hold out: you can pass your training LMDB path if you wish to double check the
performance of your model after training, or you can use it with unseen samples.

Note that by default, `predict` triggers PyTorch's inference mode, which is a specialized case where
absolutely no autograd is enabled. ``ForceRegressionTask`` uses automatic differentiation to evaluate
forces, and so for inference tasks that require gradients, you **must** pass `inference_mode=False` to
``pl.Trainer``.


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
