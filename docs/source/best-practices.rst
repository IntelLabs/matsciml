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

Point clouds to graphs
^^^^^^^^^^^^^^^^^^^^^^

``matsciml`` implements back and forth transforms to go between point cloud
and graph representations of data. Most people are interested in converting
point clouds (which are stored in the LMDB files) into graphs, which can
be done so with the ``PointCloudToGraphTransform``. The minimal configuration
is to specify the graph implementation, and optionally a cutoff parameter.
If the ``PeriodicPropertiesTransform`` is used **before** the graph transform,
the graph transform will use edges generated with consideration of the periodic
boundary conditions, in which case the cutoff parameter is not used.

A typical configuration would look like this:

.. code-block:: python
   :caption: Example PyG transformation, including periodic boundary conditions with a cutoff of 6.0.

   transforms = [
      PeriodicPropertiesTransform(6.0, adaptive_cutoff=True),
      PointCloudToGraphTransform("pyg")
   ]

This transform results in a ``graph`` key/value being added to the data
sample, and is appropriately batched depending on the implementation used.

.. autoclass:: matsciml.datasets.transforms.PointCloudToGraphTransform
   :members:


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

.. important::
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

Target normalization
^^^^^^^^^^^^^^^^^^^^

Tasks can be provided with ``normalize_kwargs``, which are key/value mappings
that specify the mean and standard deviation of a target; an example is given below.

.. code-block: python

   Task(
       ...,
       normalize_kwargs={
         "energy_mean": 0.0,
         "energy_std": 1.0,
   }
   )

The example above will normalize ``energy`` labels and can be substituted with
any of target key of interest (e.g. ``force``, ``bandgap``, etc.)

Target loss scaling
^^^^^^^^^^^^^^^^^^^

A generally common practice is to scale some targets relative to others (e.g. force over
energy, etc). To specify this, you can pass a ``task_loss_scaling``  dictionary to
any task module, which maps target keys to a floating point value that will be used
to multiply the corresponding target loss value before summation and backpropagation.

.. code-block: python
   Task(
       ...,
       task_loss_scaling={
           "energy": 1.0,
           "force": 10.0
   }
   )


A related, but alternative way to specify target scaling is to apply a *schedule* to
the training loss contributions: essentially, this provides a way to smoothly ramp
up (or down) different targets, i.e. to allow for more complex training curricula.
To achieve this, you will need to use the ``LossScalingScheduler`` callback,

.. autoclass:: matsciml.lightning.callbacks.LossScalingScheduler
   :members:


To specify this callback, you must pass subclasses of ``BaseScalingSchedule`` as arguments.
Each schedule type implements the functional form of a schedule, and currently
there are two concrete schedules. Composed together, an example would look like this

.. code-block: python

   import pytorch_lightning as pl
   from matsciml.lightning.callbacks import LossScalingScheduler
   from matsciml.lightning.loss_scaling import LinearScalingSchedule

   scheduler = LossScalingScheduler(
      LinearScalingSchedule("energy", initial_value=1.0, end_value=5.0, step_frequency="epoch")
   )
   trainer = pl.Trainer(callbacks=[scheduler])


The stepping schedule is determined during ``setup`` (as training begins), where the callback will
inspect ``Trainer`` arguments to determine how many steps will be taken. The ``step_frequency``
just specifies how often the learning rate is updated.


.. autoclass:: matsciml.lightning.loss_scaling.LinearScalingSchedule
   :members:


.. autoclass:: matsciml.lightning.loss_scaling.SigmoidScalingSchedule
   :members:


Quick debugging
^^^^^^^^^^^^^^^

Using PyTorch Lightning, the ``Trainer`` can be configured to perform
a fast "dev" run which disables checkpointing and logging, and performs
a specified number of training and validation steps for a single loop.

.. code-block:: python
   import pytorch_lightning as pl

   trainer = pl.Trainer(fast_dev_run=10)

This is an excellent way to quickly debug datasets and workflows. When
paired with ``MatSciMLDataModule.from_devset(...)``, debugging can be
significantly faster in our experience.

Diagnosing ``NaN`` during training
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``matsciml.lightning.callbacks.GradientCheckCallback`` is written to help
track down where and when parameter losses go to ``NaN``. When configured with
``verbose``, it will print out the batch index and layers where this occurs.
This callback will also zero out ``NaN`` gradients, allowing training to resume
and hoping for the problem to self-correct as the model has a chance to learn
from various training samples.

Understanding training dynamics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``matsciml.lightning.callbacks.TrainingHelperCallback`` is a callback that implements
a number of heuristics that may be helpful in getting an intuition for model
training. Some of the things this callback will watch for are embeddings that explode,
oddities with data, and making sure that gradients make it to the encoder such that
output heads are not completely ignoring the embeddings. If paired with ``wandb``
logging via ``pytorch_lightning.loggers.WandbLogger``, it will also log histograms
that help diagnose gradient behaviors.

Similarly, the ``matsciml.lightning.callbacks.ModelAutocorrelation`` callback was
inspired by observations made in LLM training research, where the breakdown of
assumptions in the convergent properties of ``Adam``-like optimizers causes large
spikes in the training loss. This callback can help identify these occurrences.

The ``devset``/``fast_dev_run`` approach detailed above is also useful for testing
engineering/infrastructure (e.g. accelerator offload and logging), but not necessarily
for probing training dynamics. Instead, we recommend using the ``overfit_batches``
argument in ``pl.Trainer``

.. code-block:: python
   import pytorch_lightning as pl

   trainer = pl.Trainer(overfit_batches=100)


This will disable shuffling in the training and validation splits (per the PyTorch Lightning
documentation), and ensure that the same batches are being reused every epoch.

.. _e3nn documentation: https://docs.e3nn.org/en/latest/

.. _IPEX installation: https://intel.github.io/intel-extension-for-pytorch/index.html#installation?platform=gpu
