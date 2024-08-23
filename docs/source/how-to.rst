How do I...
============

This page is a collection of commonly asked questions and/or patterns. If it's not here
and you're finding yourself asking it, please open a `Github issue <https://github.com/IntelLabs/matsciml/issues>`_ for discussion.

Reload a model
-------------

This can be broken up into two ways, depending on what it is you are trying to do.

If you are loading a model for inference, or someone has sent you a checkpoint,
more often than not you want `task.load_from_checkpoint`, which leverages the
method of the same name in all PyTorch Lightning ``LightningModule``s.

.. code-block:: python
   :caption: Example loading a model checkpoint from file.

   from matsciml.models import ForceRegressionTask

   task = ForceRegressionTask.load_from_checkpoint("path/to/checkpoint.ckpt")


In the other case, you want to take a **pretrained** model, and transfer the
encoder to a completely new task. Here, you want to use ``from_pretrained_encoder`` instead:

.. code-block:: python
   :caption: Taking a pretrained model and adapting it to scalar property prediction.

   from matsciml.models import ScalarRegressionTask

   task = ScalarRegressionTask.from_pretrained_encoder("path/to/checkpoint.ckpt")


Note that the behavior for multitask set ups is different.
