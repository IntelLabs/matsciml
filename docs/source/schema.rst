Schema
==========

The Open MatSciML Toolkit tries to place emphasis on reproducibility, and the general
rule of "explicit is better than implicit" by defining schema for data and other development
concepts.

The intention is to move away from hardcoded ``Dataset`` classes that are rigid in
that they require writing code, as well as not always reliably reproducible as the
underlying data and frameworks change and evolve over time. Instead, the schema
provided in ``matsciml`` tries to shift technical debt from maintaining code to
**documenting** data, which assuming a thorough and complete description, should
in principle be usable regardless of breaking API changes in frameworks that we rely
on like ``pymatgen``, ``torch_geometric``, and so on. As a dataset is being packaged
for distribution/defined, the schema should also make intentions of the developer clear
to the end-user, e.g. what target label is available, how it was calculated, and so on,
to help subsequent reproduction efforts. As an effect, this also makes development of
``matsciml`` a lot more streamlined, as it then homogenizes field names (i.e. we can
reliably expect ``cart_coords`` to be available and are cartesian coordinates).

.. TIP::
   You do not have to construct objects contained in schema if they are ``pydantic``
   models themselves: for example, the ``PeriodicBoundarySchema`` is required in
   ``DataSampleSchema``, but you can alternatively just pass a dictionary with the
   expected key/value mappings (i.e. ``{'x': True, 'y': True, 'z': False}``) for
   the relevant schema.


Dataset schema reference
########################

This schema lays out what can be described as metadata for a dataset. We define all of
the expected fields in ``targets``, and record checksums for each dataset split such
that we can record what model was trained on what specific split. Currently, it is the
responsibility of the dataset distributor to record this metadata for their dataset,
and package it as a ``metadata.json`` file in the same folder as the HDF5 files.

.. autoclass:: matsciml.datasets.schema.DatasetSchema
   :members:

Data sample schema reference
############################

This schema comprises a **single** data sample, providing standardized field names for
a host of commonly used properties. Most properties are optional for the class construction,
but we highly recommend perusing the fields shown below to find the attribute closest to
the property being recorded: ``pydantic`` does not allow arbitrary attributes to be stored
in schema, but non-standard properties can be stashed away in ``extras`` as a dictionary of
property name/values.

.. autoclass:: matsciml.datasets.schema.DataSampleSchema
   :members:
