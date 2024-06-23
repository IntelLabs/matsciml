Best practices
============

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
the tutorial outlined in the ```e3nn`` https://docs.e3nn.org/en/latest/`_ documentation,
where we use ``pymatgen`` to generate images, and for every atom in the graph,
compute nearest neighbors with some specified radius cutoff. One additional
detail we include in this approach is the ``adaptive_cutoff`` flag: if set to ``True``, will ensure
that all nodes are connected by gradually increasing the radius cutoff up
to a hard coded limit of 30 angstroms. This is intended to facilitate the
a small nominal cutoff, even if some data samples contain (intentionally)
distant atoms.

Training
----------
