Models
======

The Open MatSciML Toolkit implements a wide range of models, comprising
both baseline and state-of-the-art models. In some instances, we have implemented
the model from scratch or adapted the source from another repository, while in
others we have implemented wrappers. The advantage of one over the other is
primarily maintenance: wrappers mean we can update model definitions on the
fly and potentially access a host of models, but potentially sacrifice ease of
use. Open MatSciML Toolkit makes a best effort to try and unify interfaces,
so that every model can be used in a common pipline (e.g. datasets, callbacks),
but the edges may be sharp given the wide range of different model types we
support.

If there are models we have missed, or you encounter a bug, please open
an `issue on Github <https://github.com/IntelLabs/matsciml/issues>`_.

The model interfaces we have depend on their original implementation:
currently, we support models from DGL, PyG, and point clouds (specifically
GALA).

Implementing interfaces
-----------------------

.. note::
   This section is primarily a detail for developers. If you are a user
   and aren't interested in implementation details, feel free to skip ahead.

As mentioned earlier, models in Open MatSciML Toolkit come in two flavors:
wrappers of upstream implementations, or self-contained implementations.
Ultimately, the output of models should be standardized in one of two ways:
every model `_forward` call should return either an ``Embeddings`` (at the minimum)
or a ``ModelOutput`` object. The latter is implemented with ``pydantic`` and
therefore takes advantage of data validation workflows, including standardizing
and checking tensor shapes, which is currently the **recommended** way for model
outputs. It also allows flexibility in wrapper models to produce their own
outputs with their own algorithm, but still be used seamlessly through the pipeline.
An example of this can be found in the ``MACEWrapper``. The ``ModelOutput`` class also
includes an ``embeddings`` field, which makes it  compatible with the traditional
Open MatSciML Toolkit workflow of leveraging one or more output heads.


.. autoclass:: matsciml.common.types.Embeddings
   :members:


.. autoclass:: matsciml.common.types.ModelOutput
   :members:


.. important::
   Training tasks and workflows should branch based on the prescence of either
   objects, taking ``ModelOutput`` as the priority. For specific tasks, we can
   check if properties are set (e.g. ``total_energy`` and ``forces``), and if
   they aren't there, we should pass the ``embeddings`` to output heads.


PyG models
----------

These models require a ``PointCloudToGraphTransform`` as part of your
transform pipeline. Please see :ref:`Point clouds to graphs` to see how
to configure and add this transform.

.. autoclass:: matsciml.models.pyg.mace.MACEWrapper
   :members:


.. autoclass:: matsciml.models.pyg.EGNN
   :members:


.. autoclass:: matsciml.models.pyg.FAENet
   :members:


.. autoclass:: matsciml.models.pyg.SchNetWrap
   :members:


.. autoclass:: matsciml.models.pyg.DimeNetWrap
   :members:


.. autoclass:: matsciml.models.pyg.DimeNetPlusPlusWrap
   :members:



DGL models
----------

These models require a ``PointCloudToGraphTransform`` as part of your
transform pipeline. Please see :ref:`Point clouds to graphs` to see how
to configure and add this transform.


.. autoclass:: matsciml.models.dgl.PLEGNNBackbone
   :members:


.. autoclass:: matsciml.models.dgl.M3GNet
   :members:


.. autoclass:: matsciml.models.dgl.MEGNet
   :members:


.. autoclass:: matsciml.models.dgl.TensorNet
   :members:


.. autoclass:: matsciml.models.dgl.CHGNet
   :members:


.. autoclass:: matsciml.models.dgl.SchNet
   :members:


Point cloud models
------------------


.. autoclass:: matsciml.models.dgl.GalaPotential
   :members:
