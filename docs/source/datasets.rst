Datasets
==========

The Open MatSci ML Toolkit provides interfaces and standardizes a number of broadly
used community datasets: below is a small matrix that highlights the datasets that
are integrated into Open MatSciML Toolkit.

.. list-table:: Available datasets
   :widths: 20 10 70
   :header-rows: 1

  * - Dataset
    - License
    - Notes
  * - `Open Catalyst Project <https://github.com/Open-Catalyst-Project/ocp>`_
    - CC-BY-4.0
    - Large scale catalyst+adsorbate trajectory dataset from Meta AI research. Includes S2EF and IS2RE datasets.
  * - `Materials Project <https://materialsproject.org/>`_
    - CC-BY-4.0
    - Database of experimental and computational properties for materials relevant to diverse applications. Maintained by Lawrence Berkeley National Laboratory.
  * - `LiPS <https://archive.materialscloud.org/record/2022.45>`_
    - CC-BY-4.0
    - Small(ish) MD trajectory dataset of a Li6.75P3S11 system. Generated as part of the E(3)-GNN interatomic potential publication. Good for quick testing of energy/force regression tasks.
  * - `NOMAD <https://nomad-lab.eu/>`_
    - CC-BY-4.0
    - Refers to the data from the Novel Materials Discovery (NOMAD) laboratory, comprising a large quantity of material structures and properties derived from computational methods.
  * - `Carolina Materials Database <http://www.carolinamatdb.org/>`_
    - CC-BY-4.0
    - Dataset comprising ~200,000 materials with over 250,000 calculated properties such as formation energy and band gap.
  * - `Alexandria Database <https://alexandria.icams.rub.de/>`_
    - CC-BY-4.0
    - Pertains to the novel 2D and 3D materials database, comprising >200,000 PBE/PBESol relaxed structures with a host of calculated properties.


Users can obtain preprocessed LMDB files from the `Zenodo <https://doi.org/10.5281/zenodo.10768743>`_. Each dataset
includes predefined train/test/validation splits, which should be ready to use out of the box with the Open MatSciML Toolkit pipeline.

User reference
##############

Generally speaking, users do not have to directly interact with the underlying
``Dataset`` classes/objects unless you are doing something out of the box. At
a high-level, the general interface for all datasets includes an ``lmdb_root``
and a ``transforms`` arguments that can be configured: the former points to
a folder containing LMDB files that pertain to a particular split, and transforms
are a list of callable transformations to each data sample **after** they are loaded.
See the :ref:`Transforms` section for details.

To make use of the PyTorch Lightning abstractions, we recommend users configure
the ``MatSciMLDataModule``.

.. autoclass:: matsciml.lightning.data_utils.MatSciMLDataModule
   :members:


Dataset API reference
#####################

``BaseLMDBDataset`` reference
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. note::
   This subsection is mainly for developers.

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
