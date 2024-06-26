Datasets
==========

The Open MatSci ML Toolkit provides interfaces and standardizes a number of broadly
used community datasets:

Dataset API reference
#####################

``BaseLMDBDataset`` reference
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This class serves as the intended basis for all ``matsciml`` datasets. In addition
to functions that are in charge of reading from LMDB filetypes, the base class
implements several quality of life methods and properties that are intended to
help with high-level dataset testing and usage, such as ``save_preprocessed_data()``,
``from_devset()``, and ``sample()``.

For the developer, new datasets can be developed by inheriting from this base class,
and overriding the ``data_from_key()`` method; for most datasets, you can leverage
the base class' LMDB loading logic from this method, and in the override implement
the parsing/packaging tailored to your particular dataset::

  class NewDataset(BaseLMDBDataset):
      def data_from_key(lmdb_index, subindex):
          # returns an unpickled dictionary of data
          loaded_data = super().data_from_key(lmdb_index, subindex)
          # extract labels from unpickled data to return
          return_dict = {"targets": {"energy": loaded_data["energy"]}}
          return return_dict



.. autoclass:: matsciml.datasets.base.BaseLMDBDataset
   :members:
   :inherited-members:
