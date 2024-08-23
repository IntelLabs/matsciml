Callbacks
==========

One of the most powerful aspects of PyTorch Lightning, and by extension Open MatSciML Toolkit,
is the ability to use modular callbacks: functionality that can intercept and modify workflows
all throughout training and inference. Callbacks are called at specific events determined
by PyTorch Lightning, such as ``on_fit_start``, ``before_optimizer_step``, etc. such that
we can write procedures that target where in the pipeline certain behaviors are desirable.

.. note::
   Given the wide range of possible uses of callbacks, this page needs to be heavily
   developed further after refactoring the callback implementations.


Common callbacks and their uses
------------------------------

Some of the training-relevant callbacks are described in :ref:`Understanding training dynamics`,
and we won't duplicate them here. There are still a few useful callbacks that are particular
in when you might use them, which we will describe here.

Exponential moving average callback
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This implements the exponential moving average strategy as a callback,
using the PyTorch native implementation of EMA (i.e. not ``torch-ema``).
At a high level, the EMA callback generates a duplicate set of model
weights that are updated based on an exponential moving average; intuitively,
this leads to a smoother set of weights that have empirically been shown
to improve generalization.

The callback implements the update of the EMA weights, and tasks have been
writtten to use the EMA weights for **validation** automatically. This means
that the qualitiative behavior of training and validation loss curves will
look quite different. The callback also plugs into checkpointing: when the
model is saved, the EMA weights become *the* model weights as to not have
duplication of weights, as well as more straightforward behavior (i.e. typically
you use the save checkpoint for inference, or for re-training/fine-tuning).

.. autoclass:: matsciml.lightning.callbacks.ExponentialMovingAverageCallback
   :members:


SAM
^^^^^^^

This callback implements the "sharpness aware minimization" `technique <https://arxiv.org/abs/2010.01412>`_,
which corresponds to a family of weight-update strategies that as the name suggests,
attempts to lead to smoother model weight changes (i.e. less noisy) that are meant
to help with generalization. This makes SAM similar to the :ref:`Exponential moving average callback`,
and while the two are theoretically complimentary, we have yet to determine the
empirical benefits for choosing one over the other, or both.

The SAM implementation is controlled by two hyperparameters, ``rho`` and ``adaptive``.
The former controls the scale of conjugate updates, and ``adapative`` sets a dynamic
weighting for the conjugate weights based on the square of the weights.

.. autoclass:: matsciml.lightning.callbacks.SAM
   :members:


Callback API reference
----------------------

.. automodule:: matsciml.lightning.callbacks
   :members:
