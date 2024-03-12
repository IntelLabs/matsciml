# Contributing to Open MatSciML Toolkit

Thank you for considering a contribution to our project! We appreciate your support and collaboration in improving our software. To ensure that your contributions are valuable and well-structured, please follow the guidelines below:

## Environment

We recommend installing Open MatSciML Toolkit as an editable install with the relevant development dependencies using `pip install -e './[dev]'`.

Additionally, we ask contributors to use our `pre-commit` workflow, which can be installed via `pre-commit install`, which helps enforce code
style consistency and run static code checkers with `ruff`. Docstrings are written in [NumPy style](https://numpydoc.readthedocs.io/en/latest/format.html), and where possible,
type annotations are very helpful for users and maintainers.

## Models

Please refer to the dedicated [models writeup](./matsciml/models/README.md).

## Datasets

- Dataset contributions should include a brief description of the dataset and its available fields.
- Provide proper documentation on how to access, use, and understand the data.
- Make sure to include data preprocessing scripts if applicable.

Adding a dataset usually involves interacting with an external API to query and download data. If this is the case, a separate `{dataset}_api.py` and `dataset.py` file can be used to separate out the functionalities. In the API file, a default query can be used to save data to lmdb files, and do any initial preprocessing necessary to get the data into a usable format. Keeping track of material ID's and the status of queries.

The main dataset file should take care of all of the loading, processing and collating needed to prepare data for the training pipeline. This typically involves adding the necessary key-value pairs which are expected, such as `atomic_numbers`, `pc_features`, and `targets`.

The existing dataset's should be used as a template, and can be expanded upon depending on models needs.

## Tests

- Tests for new models and datasets should be added as necessary, following the conventions laid out for existing models and datasets.
- Follow our testing framework and naming conventions.
- Verify that all tests pass successfully before making a pull request.

Tests for each new model and datasets should be added to their respective tests folder, and follow the conventions of the existing tests. Task specific tests may be added to the model folder itself. All relevant tests must pass in order for a pull request to be accepted and merged.

Model tests may be added [here](https://github.com/IntelLabs/matsciml/tree/main/matsciml/models/dgl/tests), and dataset tests may be added to their respective dataset folders when created.

We have also defined a set of `pytest.mark`s for categorizing tests, which include:

- `pytest.mark.slow` - for compute heavy tests, which we want to avoid for CI
- `pytest.mark.remote_request` - for remote API requests, which we want to avoid for CI
- `pytest.mark.lmdb` - for LMDB/IO based actions, which can be slow with CI

Please decorate tests accordingly. If a particular case isn't captured, feel free to add new marks and append to this list.

## General Guidelines

- Make your code readable and maintainable. Use meaningful variable and function names.
- Follow the coding standards and style guidelines set in the repository.
- Include a clear and concise commit message that describes your changes.
- Ensure that your code is free of linting errors and passes code formatting checks.
- Keep your pull request focused and single-purpose. If you're addressing multiple issues, create separate pull requests for each.
- Update documentation if your contribution adds or modifies features.
- Use informative type annotations: there are some defined in `matsciml.common.types` that help express what the intended inputs are.

Once you've prepared your contribution, please submit a pull request. Our team will review it, provide feedback if needed, and work with you to merge it into the project.

__If it is your first pull request, please ensure you add your name to the [contributors list](./CONTRIBUTORS.md)!__

We appreciate your dedication to making our project better and look forward to your contributions! If you have any questions or need assistance, feel free to reach out through the issue tracker or discussions section.

Thank you for being a part of our open-source community!
