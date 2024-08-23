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
