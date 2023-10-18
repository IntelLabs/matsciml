# Contributing to Open MatSciML Toolkit

Thank you for considering a contribution to our project! We appreciate your support and collaboration in improving our software. To ensure that your contributions are valuable and well-structured, please follow the guidelines below:

## Environment

We recommend installing Open MatSciML Toolkit as an editable install with the relevant development dependencies using `pip install -e './[dev]'`.

### Models
- When contributing models, make sure they are well-documented with clear explanations and examples.
- Include a README file with model specifications, training parameters, and any relevant information.
- Code should be organized, well-commented, and follow the repository's coding style and conventions.
- If the model depends on external data or dependencies, clearly specify these requirements.

Models can come in two varieties, DGL and PyG. DGL based models generally will subclass the `BaseDGLModel` class while PyG uses `BaseModel`. 

In DGL models, the `_forward` method should be defined which is used by the underlying task's standard `forward` method. Input data should match what is expected by the base task module:
```python
    def _forward(
        self,
        graph: AbstractGraph,
        node_feats: torch.Tensor,
        pos: Optional[torch.Tensor] = None,
        edge_feats: Optional[torch.Tensor] = None,
        graph_feats: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
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
        torch.Tensor
            Model output; either embedding or projected output
        """
```

In PyG, the `_forward` method takes only the `data` object:
```python
def _forward(self, data):
```

Aside from implementing the `_forward` method of the model itself, the constituent building blocks should be broken up into their own files, respective to what their functions are. For example, layer based classes and utilities should be placed into a `layers.py` file, and other helpful functions can be placed in a `helper.py` or `utils.py` file. 

Completed models can be added to the list of imports in `./matsciml/models/dgl/__init__.py`.

### Datasets
- Dataset contributions should include a brief description of the dataset and its available fields.
- Provide proper documentation on how to access, use, and understand the data.
- Make sure to include data preprocessing scripts if applicable.

Adding a dataset usually involves interacting with an external API to query and download data. If this is the case, a separate `{dataset}_api.py` and `dataset.py` file can be used to separate out the functionalities. In the API file, a default query can be used to save data to lmdb files, and do any initial preprocessing necessary to get the data into a usable format. Keeping track of material ID's and the status of queries. 

The main dataset file should take care of all of the loading, processing and collating needed to prepare data for the training pipeline. This typically involves adding the necessary key-value pairs which are expected, such as `atomic_numbers`, `pc_features`, and `targets`.

The existing dataset's should be used as a template, and can be expanded upon depending on models needs.


### Tests
- Tests for new models and datasets should be added as necessary, following the conventions laid out for existing models and datasets.
- Follow our testing framework and naming conventions.
- Verify that all tests pass successfully before making a pull request.

Tests for each new model and datasets should be added to their respective tests folder, and follow the conventions of the existing tests. Task specific tests may be added to the model folder itself. All relevant tests must pass in order for a pull request to be accepted and merged. 

Model tests may be added [here](https://github.com/IntelLabs/matsciml/tree/main/matsciml/models/dgl/tests), and dataset tests may be added to their respective dataset folders when created.

### General Guidelines
- Make your code readable and maintainable. Use meaningful variable and function names.
- Follow the coding standards and style guidelines set in the repository.
- Include a clear and concise commit message that describes your changes.
- Ensure that your code is free of linting errors and passes code formatting checks.
- Keep your pull request focused and single-purpose. If you're addressing multiple issues, create separate pull requests for each.
- Update documentation if your contribution adds or modifies features.

Once you've prepared your contribution, please submit a pull request. Our team will review it, provide feedback if needed, and work with you to merge it into the project.

We appreciate your dedication to making our project better and look forward to your contributions! If you have any questions or need assistance, feel free to reach out through the issue tracker or discussions section.

Thank you for being a part of our open-source community!