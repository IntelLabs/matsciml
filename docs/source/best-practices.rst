Best practices
==============

Thanks to the flexibility of the Open MatSciML Toolkit, there is a need
to document regular usage patterns, or what one may consider as "best practices".

General concepts
----------------

Periodic boundary conditions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Most of the datasets in ``matsciml`` contain periodic/crystal structures.
While there is yet to be a unique data structure/featurization method that
holistically describes periodicity, one of the most common strategies is
to wire graph edges in a way that mimics neighboring cell connectivity.

The way this is implemented in ``matsciml`` is to include the transform,
``PeriodicPropertiesTransform``:

.. autofunction:: matsciml.datasets.transforms.PeriodicPropertiesTransform

This implementation is heavily based off
the tutorial outlined in the `e3nn documentation`_ where we use ``pymatgen``
to generate images, and for every atom in the graph,
compute nearest neighbors with some specified radius cutoff. One additional
detail we include in this approach is the ``adaptive_cutoff`` flag: if set to ``True``, will ensure
that all nodes are connected by gradually increasing the radius cutoff up
to a hard coded limit of 100 angstroms. This is intended to facilitate the
a small nominal cutoff, even if some data samples contain (intentionally)
significantly more distant atoms than the average sample. By doing so, we
improve computational efficiency by not needing to consider many more edges
than required.

Accelerator usage
-----------------

Since Open MatSciML Toolkit utilizes PyTorch Lightning, most concepts
(i.e. models, datamodules) are written in the abstract, as ``Trainer``
handles all of the host-accelerator data transfers and computation::

  import pytorch_lightning as pl
  import matsciml

  # configure everything else above
  trainer = pl.Trainer(accelerator="xpu")

In addition to accelerators supported in PyTorch Lightning such
as Nvidia GPUs, ``matsciml`` also includes support for GPUs from
the Intel®️ Data Center GPU Max Series. Client grade GPUs from Intel®️
have not been tested yet, but presumably should work with the same
abstractions.

Intel®️ XPU usage
^^^^^^^^^^^^^^^^

Currently, in order to use Intel GPUs with PyTorch and ``matsciml``,
the XPU version of ``intel_extension_for_pytorch`` (IPEX) must be installed
alongside oneAPI with supported video drivers and compute runtimes.
The current version pinned in ``matsciml`` is PyTorch 2.1.0, and
``intel_extension_for_pytorch==2.1.10+xpu``. Please consult the `IPEX installation`_
 documentation for step-by-step installation instructions; ``matsciml``
 provides a ``conda-xpu.yml`` environment specification that can be used
 by ``conda`` or ``mamba`` to install everything *except* oneAPI and GPU drivers.


Training
--------

.. _e3nn documentation: https://docs.e3nn.org/en/latest/

.. _IPEX installation: https://intel.github.io/intel-extension-for-pytorch/index.html#installation?platform=gpu
