Developers guide
===================

There are a lot of ways to get your hands dirty contributing to the Open MatSciML Toolkit,
ranging from adding/fixing documentation, bug fixes, to new features, models, and/or datasets.
If you are new to open source/Github, here is a `link <https://laserkelvin.github.io/blog/2021/10/contributing-github/>`_ to a post
written for the more general scientist, but an abridged version of the workflow is given below;
we assume you are working in a terminal, and if you use a graphical interface for controlling
``git`` (e.g. VSCode, the Github client itself) you should refer to their documentation for
specifics but the general idea should remain the same.

1. Make sure you have an account on Github, and that your local environment has ``git`` set up and configured with your Github credentials for SSH access. See `here <https://docs.github.com/en/authentication/connecting-to-github-with-ssh>`_ for instructions to do so.
2. At the `public repository <https://github.com/IntelLabs/matsciml`_, create a fork. When that's done, clone *your fork* locally where you normally keep code.
3. Navigate to your freshly cloned ``matsciml``; by default you will be on ``main`` branch.
4. We recommend making changes on new branches: you can create a new branch by running ``git checkout -b <name-of-branch>``. We recommend naming the branch something descriptive, and optionally if you are addressing an issue, prefix the name of the branch with the Github issue number.
5. Install ``matsciml`` in a virtual environment, whether that be with ``conda``/``mamba``, or with ``venv``. Be sure to install it in editable mode (``-e``), with the development requirements (``./[dev]``).
6. Run ``pre-commit install``: this will set up all the ``git`` hooks that execute *before* commits to catch any mistakes, format code changes, etc.
7. Start making changes! We **highly** recommend making small changes, and committing them frequently. Minimizing the amount of changes between commits, and in general in pull requests, helps the review process substantially.
8. Once you get to a point where you are happy with enough changes, run ``git push -u origin <name-of-branch>`` to push your *local* changes to *your fork*. This command assumes ``origin`` is the name of the remote, which if you followed the instructions, should be the case.
9. Open your browser and head to your fork's repository: there should be a pop-up now indicating that you have new changes that could be submitted as a pull request.
10. Submit a pull request: try and be descriptive and succinct in what you've done in your branch. If there is an issue tied to this, add "This closes #<issue-number>" to automatically link the issue to your pull request.
11. The maintainers will review the code, and you'll work together to figure out how it can be merged!
12. Once the changes have been approved, if it's your first pull request we'll ask you to add your name to the contributors list! You can make this as another local commit, then run ``git push`` again.

General Guidelines
------------------

* Make your code readable and maintainable. Use meaningful variable and function names.
* Follow the coding standards and style guidelines set in the repository.
* Include a clear and concise commit message that describes your changes.
* Ensure that your code is free of linting errors and passes code formatting checks.
* Keep your pull request focused and single-purpose. If you're addressing multiple issues, create separate pull requests for each.
* Update documentation if your contribution adds or modifies features.
* Use informative type annotations: there are some defined in ``matsciml.common.types`` that help express what the intended inputs are.

We appreciate your dedication to making our project better and look forward to your contributions! If you have any questions or need assistance, feel free to reach out through the issue tracker or discussions section.

Thank you for being a part of our open-source community!


Specific instructions
---------------------

Documentation
^^^^^^^^^^^^^

The hosted documentation uses ``sphinx`` for building and ``readthedocs`` for CI building and hosting. Documentation
is written in `reStructured text <https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html>`_, which is similar
but different to Markdown syntax which most are familiar with.

Documentation files can be found in ``docs/source`` in the ``matsciml`` root folder. In the ``docs`` folder there is a ``Makefile`` which,
if you have the necessary development packages installed, can be used to build the documentation locally for viewing *before* submitting
PRs. Simply navigate into ``docs``, then run ``make html`` that will then build into ``docs/html`` for you to open up in a browser.
We recommend you make sure that the documentation is formatted as intended before submitting.

Models
^^^^^^

Please refer to the dedicated `models writeup <https://github.com/IntelLabs/matsciml/models/README.md>`_ until we migrate the documentation here.

Datasets
^^^^^^^^

* Dataset contributions should include a brief description of the dataset and its available fields.
* Provide proper documentation on how to access, use, and understand the data.
* Make sure to include data preprocessing scripts if applicable.

Adding a dataset usually involves interacting with an external API to query and download data. If this is the case, a separate ``{dataset}_api.py`` and ``dataset.py`` file can be used to separate out the functionalities. In the API file, a default query can be used to save data to lmdb files, and do any initial preprocessing necessary to get the data into a usable format. Keeping track of material ID's and the status of queries.

The main dataset file should take care of all of the loading, processing and collating needed to prepare data for the training pipeline. This typically involves adding the necessary key-value pairs which are expected, such as ``atomic_numbers``, ``pc_features``, and ``targets``.

The existing dataset's should be used as a template, and can be expanded upon depending on models needs.

Tests
^^^^^

* Tests for new models and datasets should be added as necessary, following the conventions laid out for existing models and datasets.
* Follow our testing framework and naming conventions.
* Verify that all tests pass successfully before making a pull request.

Tests for each new model and datasets should be added to their respective tests folder, and follow the conventions of the existing tests. Task specific tests may be added to the model folder itself. All relevant tests must pass in order for a pull request to be accepted and merged.

Model tests may be added `here <https://github.com/IntelLabs/matsciml/tree/main/matsciml/models/dgl/tests>`_, and dataset tests may be added to their respective dataset folders when created.

We have also defined a set of `pytest.mark`s for categorizing tests, which include:

* ``pytest.mark.slow`` - for compute heavy tests, which we want to avoid for CI
* ``pytest.mark.remote_request`` - for remote API requests, which we want to avoid for CI
* ``pytest.mark.lmdb`` - for LMDB/IO based actions, which can be slow with CI

Please decorate tests accordingly. If a particular case isn't captured, feel free to add new marks and append to this list.
