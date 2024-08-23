Transforms
==========

The design philosophy behind the Open MatSci ML Toolkit pipeline focuses on
modularity: the behavior of each part of the pipeline can be modified
without necessarily negatively impacting others, and the ``transform``
interface was designed to allow for user flexibility with data without
needing to modify dataset codes.

The Open MatSciML Toolkit implements a number of transformations for
a number of diverse use cases. Some are model specific, such as the
frame averaging technique, which preprocesses data specific to these
models in a way that means we don't have to maintain them for every
single dataset. Others are for tasks, such as noisy nodes for pretraining,
which will process atomic coordinates in preparation for the training task.
Most generally, we provide transformations between point cloud representations,
and graph implementations by DGL and PyG. For users looking for instructions
on how to use this functionality, please see :ref:`Best practices`.

Abstraction
###########

.. note::

   This explanation is more intended for developers, and/or those seeking
   to understand how the pipeline is composed.

The basis for the transform pipeline is in ``AbstractDataTransform``, which
structures all concrete transforms to have ``setup_transform`` and ``__call__``
methods. This design is heavily influenced by how PyTorch Lightning behaves,
and the former is called by the dataset object to provide a way to access
dataset variables ahead of execution. The latter is an abstract method that
implements the actual behavior of the transformation, taking in a data sample,
transforms it, and then returns the result for the next transform stage. The
core idea is to allow multiple transforms to be chained together, in a
permutation invariant way (i.e. first transform in works first).

Transform API reference
#######################

.. autosummary::
   :toctree: generated
   :recursive:

   matsciml.datasets.transforms
