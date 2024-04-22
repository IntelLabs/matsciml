Transforms
==========

The design philosophy behind the Open MatSci ML Toolkit pipeline focuses on
modularity: the behavior of each part of the pipeline can be modified
without necessarily negatively impacting others, and the ``transform``
interface was designed to allow for user flexibility with data without
needing to modify dataset codes.

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
core idea is to allow
