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

.. warning::
   The XPU packaging ecosystem is currently being overhauled substantially,
   notably from PyTorch 2.4.0 onwards, should have native support. As such,
   a lot of information in this section will soon become out of date in
   that it won't be absolutely necessary to install IPEX in order to use
   Intel XPUs.

Currently, in order to use Intel GPUs with PyTorch and ``matsciml``,
the XPU version of ``intel_extension_for_pytorch`` (IPEX) must be installed
alongside oneAPI with supported video drivers and compute runtimes.
The current version pinned in ``matsciml`` is PyTorch 2.1.0, and
``intel_extension_for_pytorch==2.1.10+xpu``. Please consult the `IPEX installation`_
documentation for step-by-step installation instructions; ``matsciml``
provides a ``conda-xpu.yml`` environment specification that can be used
by ``conda`` or ``mamba`` to install everything *except* oneAPI and GPU drivers.

The Lightning accelerator and strategies can be found in ``matsciml.lightning.xpu``.
Normally, these are automatically registered to Lightning by importing
``matsciml`` and so no further action is required. If needed, however,
the :py:func:`matsciml.lightning.xpu.XPUAccelerator` object can be passed
directly to the ``accelerator`` argument in ``Trainer``.

.. autofunction:: matsciml.lightning.xpu.SingleXPUStrategy

The `xpu_example.py <https://github.com/IntelLabs/matsciml/tree/main/examples/devices/xpu_example.py>`_ provides
an example for XPU usage.

.. important::
   Newer versions (>2.0.0) of IPEX and oneAPI treats compute tiles as independent
   devices. By specifying ``accelerator=xpu`` in ``Trainer`` as shown in the
   ``xpu_example.py`` script, only one tile per physical card will be used,
   even if multiple tiles are present. The Intel Data Center GPU Max Series 1550
   contains *two* tiles and will require distributed data parallelism *or* additional
   configuration to utilize both compute tiles.


Distributed training
^^^^^^^^^^^^^^^^^^^^

Similar to accelerator usage, distributed data parallelism is configured using
PyTorch Lightning abstractions by specifying a ``strategy`` in ``Trainer``.

For Lightning supported accelerators, consult the Lightning documentation for
how to configure strategies for distributed data parallelism.

For Intel CPUs and XPUs, ``matsciml`` provides the :py:func:`matsciml.lightning.ddp.MPIEnvironment`
and :py:func:`matsciml.lightning.ddp.MPIDDPStrategy` classes that inherit from ``LightningEnvironment``
and ``DDPStrategy`` respectively. Effectively, these classes
inform ``Trainer`` how to parse environment variables for information about
distributed workers (i.e. rank and world size) from Intel®️ MPI, and to wrap the model and
data samplers in their respective distributed classes. To use these
classes, processes must be launched by wrapping the ``python`` script execution
with ``mpirun``::

  mpirun -n 4 -map-by socket python <script to run>.py

This will launch four ranks on one node, with memory binding to sockets
to minimize cross-socket memory traffic which negatively impacts performance.

For multiple nodes, a text file containing hostnames of each compute
node can be passed to ``mpirun``::

  mpirun -n 16 -ppn 4 -map-by socket -f hosts.txt python <script to run>.py

This configures four ranks per node, and with a total of 16 ranks, equates
to expecting four nodes specified in a file called ``hosts.txt``. If you are
using a scheduler like Slurm, the file can be generated as part of your batch
script, either by reading from environment variables, or with ``scontrol`` and
similar tools. If Slurm has been properly configured, the Lightning ``SlurmEnvironment``
can also be used; the ``MPIEnvironment`` in ``matsciml`` is intended mainly for
bare-metal or lower level usage. This functionality is wrapped by a custom
strategy:

.. autofunction:: matsciml.lightning.ddp.MPIDDPStrategy

All arguments for this class are optional. For CPU backend, using the ``oneccl_bindings_for_pytorch`` package for communications,
a variant of the strategy is registered with ``ddp_with_ccl``::

  import pytorch_lightning as pl
  from matsciml.lightning import ddp

  trainer = pl.Trainer(strategy="ddp_with_ccl", devices=4, num_nodes=4)

.. important::
   This configuration will distribute the full dataset across each rank.
   What this means is the ``batch_size`` set in datamodules specifies the
   value *per rank*, i.e. in the example above, if ``batch_size`` is set
   to 32, then the *effective* batch size will be $(32 x 4 ranks x 4 nodes)$.
   For scaling experiments, the base learning rate is typically scaled by
   square root of the total number of workers (the world size) to match
   the variance.

For Intel XPUs, everything above applies, with the exception that ``ddp_with_ccl``
should be replaced with ``ddp_with_xpu``, which uses the ``XPUAccelerator`` as
the accelerator.

Training
--------

.. _e3nn documentation: https://docs.e3nn.org/en/latest/

.. _IPEX installation: https://intel.github.io/intel-extension-for-pytorch/index.html#installation?platform=gpu
