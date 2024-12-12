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


Creating datasets with schema
#############################

.. NOTE::
   This section is primarily for people interested in developing new datasets.

The premise behind defining these schema rigorously is to encourage reproducible workflows
with (hopefully) less technical debt: we can safely rely on data to validate itself,
catch mistakes when serializing/dumping data for others to use, and set reasonable expectations
on what data will be available at what parts of training and evaluation. For those interested
in creating their own datasets, this section lays out some preliminary notes on how to wrangle
your data to adhere to schema, and make the data ready to be used by the pipeline.

Matching your data to the schema
================================

First step in dataset creation is taking whatever primary data format you have, and mapping
them to the ``DataSampleSchema`` laid out above. The required keys include ``index``, ``cart_coords``,
and so on, and by definition need to be provided. The code below shows an example loop
over a list of data, which we convert into a dictionary with the same keys as expected in ``DataSampleSchema``:

.. ::
 :caption: Example abstract code for mapping your data to the schema
   all_data = ...  # should be a list of samples
   samples = []
   for index, data in enumerate(all_data):
       temp_dict = {}
       temp_dict["cart_coords"] = data.positions
       temp_dict['index'] = index
       temp_dict['datatype'] = "OptimizationCycle"   # must be one of the enums
       temp_dict['num_atoms'] = len(data.positions)
       schema = DataSampleSchema(**temp_dict)
       samples.append(schema)

You end up with a list of ``DataSampleSchema`` which undergo all of the validation
and consistency checks.

Data splits
======================

At this point you could call it a day, but if we want to create uniform random
training and validation splits, this is a good point to do so. The code below
shows one way of generating the splits: keep in mind that this mechanism for
splitting might not be appropriate for your data - to mitigate data leakage,
you may need to consider using more sophisticated algorithms that consider chemical
elements, de-correlate dynamics, etc. Treat the code below as boilerplate, and
modify it as needed.

.. ::
   :caption: Example code showing how to generate training and validation splits
   import numpy as np
   import h5py
   from matsciml.datasets.generic import write_data_to_hdf5_group

   SEED = 73926   # this will be reused when generating the metadata
   rng =  np.random.default_rng(SEED)

   all_indices = np.arange(len(samples))
   val_split = int(len(samples) * 0.2)
   rng.shuffle(all_indices)
   train_indices = all_indices[val_split:]
   val_indices = all_indices[:val_split]

   # instantiate HDF5 files
   train_h5 = h5py.File("./train.h5", mode="w")

   for index in train_indices:
        sample = samples[index]
        # store each data sample as a group comprising array data
        group = train_h5.create_group(str(index))
        # takes advantage of pydantic serialization
        for key, value in sample.model_dump(round_trip=True).items():
            if value is not None:
                write_data_to_hdf5_group(key, value, group)


Repeat the loop above for your validation set.

Dataset metadata
==================

Once we have created these splits, there's a bunch of metadata associated
with **how** we created the splits that we should record so that at runtime,
there's no ambiguity which data and splits are being used and where they
came from.

.. ::
   from datetime import datetime
   from matsciml.datasets.generic import MatSciMLDataset
   from matsciml.datasets.schema import DatasetSchema

   # use the datasets we created above; `strict_checksum` needs to be
   # set the False here because we're going to be generating the checksum
   train_dset = MatSciMLDataset("./train.h5", strict_checksum=False)
   train_checksum = train_dset.blake2s_checksum

   # fill in the dataset metadata schema
   dset_schema = DatasetSchema(
       name="My new dataset",
       creation=datetime.now(),
       split_blake2s={
           "train": train_checksum,
           "validation": ...,
           "test": ...,  # these should be made the same way as the training set
       },
       targets=[...],   # see below
       dataset_type="OptimizationCycle",   # choose one of the `DataTypeEnum`
       seed=SEED,    # from the first code snippet
   )
   # writes the schema where it's expected
   dset.to_json_file("metadata.json")


Hopefully you can appreciate that the metadata is meant to lessen the burden
of future users of the dataset (including yourself!). The last thing to cover
here is that ``targets`` was omitted in the snippet above: this field is meant
for you to record every property that may or may not be part of the standard
``DataSampleSchema`` which is intended to be used throughout training. This is
the ``TargetSchema``: you must detail the name, expected shape, and a short
description of every property (including the standard ones). The main motivation
for this is that ``total_energy`` for one dataset may mean something very different
between one dataset to the next (electronic energy? thermodynamic corrections?),
and specifying this for the end user will remove any ambiguities.


.. autoclass:: matsciml.datasets.schema.TargetSchema
   :members:
