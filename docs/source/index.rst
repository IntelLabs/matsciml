.. Open MatSciML Toolkit documentation master file, created by
   sphinx-quickstart on Fri Apr 12 22:42:32 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

The Open MatSciML Toolkit
=================================================

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   Getting started <self>
   datasets
   transforms
   models
   training
   callbacks
   experiment-interface
   best-practices
   how-to
   developers

The Open MatSciML Toolkit comprises a framework and benchmark suite for AI-accelerated materials discovery,
with a strong emphasis on a low barrier to entry and continuity across laptops to high performance computing clusters.
The toolkit provides access to state-of-the-art model architectures like MACE, FAENet, and GemNet,
coupled with a broad range of datasets including but not limited to `Open Catalyst`_, the `Materials Project`_,
and `NOMAD`_, and alongside an end-to-end modular pipeline for experimentation and development.

.. _Open Catalyst: https://github.com/Open-Catalyst-Project/ocp
.. _Materials Project: https://next-gen.materialsproject.org/
.. _NOMAD: https://nomad-lab.eu/

Getting started
----------------

If you are looking to get started with working on the Open MatSciML Toolkit, we highly recommend
you peruse the documentation. A good place to start is seeing what :ref:`Datasets` and :ref:`Models`
are supported by the toolkit. If any pique your interest, check out the `Github examples <https://github.com/IntelLabs/matsciml/tree/main/examples>`_
folder, which can be a good place to get up and running. Once you've got your teeth sunk into the
general functionality of the toolkit, check out :ref:`Best practices` for additional functionality
of the toolkit. If you're stuck, please feel free to open an issue or start a discussion on Github!
When you're ready, checkout the :ref:`Developers guide` to see how you can contribute to the toolkit!

Installation
^^^^^^^^^^^^^

This section is a work in progress, for now we defer you to the `Github readme <https://github.com/IntelLabs/matsciml/tree/main?tab=readme-ov-file#installation>`_
for installation instructions.


Citation
---------

If you use Open MatSci ML Toolkit for your research and/or production, we request you provide a citation with the following:

.. code-block:: bibtex

  @article{openmatscimltoolkit,
  title = {The Open {{MatSci ML}} Toolkit: {{A}} Flexible Framework for Machine Learning in Materials Science},
  author = {Miret, Santiago and Lee, Kin Long Kelvin and Gonzales, Carmelo and Nassar, Marcel and Spellings, Matthew},
  year = {2023},
  journal = {Transactions on Machine Learning Research},
  issn = {2835-8856}
  }

Additionally, please cite the original publications for the datasets used. See the ":doc:`datasets`" page for references.


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
