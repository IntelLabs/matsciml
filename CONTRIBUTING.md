# Contributing to Open MatSciML Toolkit

Thank you for considering a contribution to our project! We appreciate your support and collaboration in improving our software. To ensure that your contributions are valuable and well-structured, please follow the guidelines below:

## Environment

We recommend installing Open MatSciML Toolkit as an editable install with the relevant development dependencies using `pip install -e './[dev]'`.

Additionally, we ask contributors to use our `pre-commit` workflow, which can be installed via `pre-commit install`, which helps enforce code
style consistency and run static code checkers such as `flake8` and `bandit`. For the latter, we rely on `black` to do the vast majority
of code formatting. While not explicitly enforced, we highly encourage the use of type annotations for function arguments in addition
to docstrings in NumPy style.

## Models

At a high level, model implementations in Open MatSciML are expected to emit a system-level embedding; i.e. $B$
vector embeddings with dimension $D$ for $B$ minibatch size, and $D$ some hyperparameter of your model, given
some material structure comprising node, edge, and graph features.

- When contributing models, make sure they are well-documented with clear explanations and examples.
- Include a README file with model specifications, training parameters, and any relevant information.
- Code should be organized, well-commented, and follow the repository's coding style and conventions.
- If the model depends on external data or dependencies, clearly specify these requirements.

While not compulsory, it is strongly recommended to inherit from one of the three abstract classes in `matsciml.models.base`
depending on the type of data representation being used and backend: `AbstractPointCloudModel`, `AbstractDGLModel`, and
`AbstractPyGModel`. The two main utilities of these classes are: (1) provide the same embedding table basis (e.g. for nodes
and edges) for consistency, and (2) to abstract out common functions, particularly how to "read" data from the minibatches.
As an example, DGL and PyG have different interfaces, and by inheriting from the right parent class, ensures that the correct
features are mapped to the right arguments, etc. If your abstraction permits, the `_forward` method should be the only method
you need to override, which returns the system/graph and point/node-level embeddings in a standardized data structure:

```python
    def _forward(
        self,
        graph: AbstractGraph,
        node_feats: torch.Tensor,
        pos: Optional[torch.Tensor] = None,
        edge_feats: Optional[torch.Tensor] = None,
        graph_feats: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Embeddings:
        """
        Sets args/kwargs for the expected components of a graph-based
        model. At the bare minimum, we expect some kind of abstract
        graph structure, along with tensors of atomic coordinates and
        numbers to process. Optionally, models can include edge and graph
        features, but is left for concrete classes to implement how
        these are obtained.

        Parameters
        ----------
        graph : AbstractGraph
            Graph structure implemented in a particular framework
        node_feats : torch.Tensor
            Atomic numbers or other featurizations, typically shape [N, ...] for N nuclei
        pos : Optional[torch.Tensor]
            Atom positions with shape [N, 3], by default None to make this optional
            as some architectures may pass them as 'node_feats'
        edge_feats : Optional[torch.Tensor], optional
            Edge features to process, by default None
        graph_feats : Optional[torch.Tensor], optional
            Graph-level attributes/features to use, by default None

        Returns
        -------
        matsciml.common.types.Embeddings
            Data structure containing system/graph and point/node-level embeddings.
        """
```

The `**kwargs` ensures that any additional variables that are not required by your architecture (for example graph features)
are not needed as explicit arguments.

If variables/features are required by the model, one can override the `read_batch` method. See the [MPNN](https://github.com/IntelLabs/matsciml/blob/main/matsciml/models/dgl/mpnn.py)
wrapper to see how this pattern can be used to check for data within a batch.

Aside from implementing the `_forward` method of the model itself, the constituent building blocks should be broken up into their own files, respective to what their functions are. For example, layer based classes and utilities should be placed into a `layers.py` file, and other helpful functions can be placed in a `helper.py` or `utils.py` file.

Completed models can be added to the list of imports in `./matsciml/models/<framework>/__init__.py`, where `<framework>` can be `dgl` or `pyg`.

As a general note, it is recommended to try and separate components from the model architecture. As an example, if you are developing a
graph architecture called `AmazingModel`, if possible implement an `AmazingModelConv` block that details the message passing logic, while
an overarching `AmazingModel` (inheriting from `AbstractDGLModel` or `AbstractPyGModel`) _composes_ multiple blocks together:

```python
class AmazingModelConv(MessagePassing):
    def message(...):
        ...

    def aggregate(...):
        ...

    def forward(self, graph, node_feats, edge_feats):
        messages = self.message(graph, node_feats, edge_feats)
        new_node_feats = self.aggregate(graph, messages)
        return new_node_feats


class AmazingModel(AbstractPyGModel):
    def _forward(...):
        for block in self.blocks:
            node_feats = block(graph, node_feats, edge_feats)
        graph_feats = readout(node_feats)
        return graph_feats
```

### DGL models

DGL does not provide a class to inherit from for the message passing step, and instead, relies
on users to define user-defined functions (`udf`), and extensive use of graph scopes.

We recommend reviewing the [MPNN](https://github.com/IntelLabs/matsciml/blob/main/matsciml/models/dgl/mpnn.py) wrapper
to see a simplified case, and the [MegNet](https://github.com/IntelLabs/matsciml/tree/main/matsciml/models/dgl/megnet) implementation
for a more complex case.

### PyG models

Ideally, message passing layers implemented in PyG should inherit from the `torch_geometric.nn.MessagePassing` class, and implement
the corresponding `message`, `propagate`, etc. functions as appropriate.

### Point cloud models

Models that operate on point clouds directly are not necessarily as complex as graph architectures,
as they do not need to rely on framework abstractions to perform message passing. Tt is still best
to make model architectures as modular as possible, but we do not have any rigorous style enforcement
for this type of model.

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
