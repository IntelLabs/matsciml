Training pipeline
=================

Task abstraction
================

The Open MatSciML Toolkit uses PyTorch Lightning abstractions for managing the flow
of training: how data from a datamodule gets mapped, to what loss terms are calculated,
to what gets logged is defined in a base task class. From start to finish, this module
will take in the definition of an encoding architecture (through ``encoder_class`` and
``encoder_kwargs`` keyword arguments), construct it, and in concrete task implementations,
initialize the respective output heads a set of provided or task-specific target keys.
The ``encoder_kwargs`` specification makes things a bit more verbose, but this ensures
that the hyperparameters are saved appropriately per the ``save_hyperparameters`` method
in PyTorch Lightning.


``BaseTaskModule`` API reference
--------------------------------

.. autoclass:: matsciml.models.base.BaseTaskModule
   :members:


Multi task reference
--------------------------------

One core functionality for ``matsciml`` is the ability to compose multiple tasks
together, in an (almost) seamless fashion from the single task case.

.. important::
   The ``MultiTaskLitModule`` is not written in a particularly friendly way at
   the moment, and may be subject to a significant refactor later!


.. autoclass:: matsciml.models.base.MultiTaskLitModule
   :members:


``OutputHead`` API reference
----------------------------

While there is a singular ``OutputHead`` definition, the blocks that constitute
an ``OutputHead`` can be specified depending on the type of model architecture
being used. The default stack is based on simple ``nn.Linear`` layers, however,
for architectures like MACE which may depend on preserving irreducible representations,
the ``IrrepOutputBlock`` allows users to specify transformations per-representation.

.. autoclass:: matsciml.models.common.OutputHead
   :members:


.. autoclass:: matsciml.models.common.OutputBlock
   :members:


.. autoclass:: matsciml.models.common.IrrepOutputBlock
   :members:


Task API reference
##################

Scalar regression
-----------------

This task is primarily designed for tasks adjacent to property prediction: you can
predict an arbitrary number of properties (per output head), based on a shared
embedding (i.e. one structure maps to a single embedding, which is used by each head).

A special case for using this class would be in tandem (as a multitask setup) with
the :ref:`_gradfree_force`, which treats energy/force prediction as two
separate output heads, albeit with the same shared embedding.

Please use continuous valued (e.g. ``nn.MSELoss``) loss metrics for this task.


.. autoclass:: matsciml.models.base.ScalarRegressionTask
   :members:


Binary classification
-----------------------

This task, as the name suggests, uses the embedding to perform one or more binary
classifications with a shared embedding. This can be something like a ``stability``
label like in the Materials Project. Keep in mind, however, that a special class
exists for crystal symmetry classification.

.. _crystal_symmetry:

Crystal symmetry classification
-------------------------------

This task is a specialized class for what is essentially multiclass classification,
where given an embedding, we predict which crystal space group the structure belongs
to using ``nn.CrossEntropyLoss``. This can be a good potential pretraining task.


.. note::
   This task expects that your data includes ``spacegroup`` target key.

.. autoclass:: matsciml.models.base.CrystalSymmetryClassificationTask
   :members:


Force regression task
---------------------

This task implements energy/force regression, where an ``OutputHead`` is used to first
predict the energy, followed by taking its derivative with respect to the input coordinates.
From a developer perspective, this task is quite mechanically different due to the need
for manual ``autograd``, which is not normally supported by PyTorch Lightning workflows.


.. note::
   This task expects that your data includes ``force`` target key.

.. autoclass:: matsciml.models.base.ForceRegressionTask
   :members:


.. _gradfree_force:

Gradient-free force regression task
-----------------------------------

This task implements a force prediction task, albeit as a direct output head property
prediction as opposed to the derivative of an energy value using ``autograd``.

.. note::
   This task expects that your data includes ``force`` target key.

.. autoclass:: matsciml.models.base.GradFreeForceRegression
   :members:


Node denoising task
-------------------

This task implements a powerful, and recently becoming more popular, pre-training strategy
for graph neural networks. The premise is quite simple: an encoder learns as a denoising
autoencoder by taking in a perturbed structure, and attempting to predict the amount of
noise in the 3D coordinates.

As a requirement, this task requires the following data transform; you are able to specify
the scale of the noise added to the positions and intuitively the large the scale, the higher
potential difficulty in the task.

.. autoclass:: matsciml.datasets.transforms.pretraining.NoisyPositions
   :members:


.. autoclass:: matsciml.models.base.NodeDenoisingTask
   :members:
