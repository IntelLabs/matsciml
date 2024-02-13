# Model implementations

At a high level, model implementations in Open MatSciML are expected to emit a system-level $[B, D]$ embedding,
where $B$ is the minibatch size and $D$ is some model (typically `output_dim`) hyperparameter given
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

If variables/features are required by the model, one can override the `read_batch` method. See the [MPNN](https://github.com/IntelLabs/matsciml/blob/main/matsciml/models/dgl/mpnn.py) wrapper to see how this pattern can be used to check for data within a batch. *An important note*: the recommended style is to have tensor creation as a __transform__, rather than implement it in `read_batch`; PyTorch Lightning will automatically move data samples to the correct device, whereas if including it in `read_batch` will require explicit data movement which is a frequently made error.

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
    def _forward(...) -> Embeddings:
        for block in self.blocks:
            node_feats = block(graph, node_feats, edge_feats)
        graph_feats = readout(node_feats)
        return Embeddings(system_embedding=graph_feats, point_embedding=node_feats)
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
as they do not need to rely on framework abstractions to perform message passing. It is still best
to make model architectures as modular as possible, but we do not have any rigorous style enforcement
for this type of model.

## Testing

Testing is a task that is necessary to make sure models implemented in Open MatSciML Toolkit are functional
at the fullest extent: we want to try and avoid architectures implemented for specific datasets, instead
ideally we want to preserve free composition across data, models, and tasks.

We plan to implement a set of unified unit tests, but for now we require model contributions to come with
their own minimal unit test suite to test functionality. For a clean example, see [`matsciml/models/pyg/tests/test_egnn.py`][egnn-test],
but we will reproduce the relevant bits below. The two main test design targets are:

1. Ensure your new architecture can read in data from "all" datasets
2. Ensure model outputs are: real, not `NaN`, are finite, and are within reasonable ranges (e.g. not 10 million)

For a hypothetical new model, `NewModel` implemented with PyTorch Geometric, the
test below automatically tests against datasets contained in the registry,
loads a batch, then runs the data through the model. To adapt it to your own
architecture, you nominally just need to swap out `model_fixture`.

```python
import pytest

from matsciml import datasets
from matsciml.datasets.transforms import PeriodicPropertiesTransform, PointCloudToGraphTransform
from matsciml.lightning import MatSciMLDataModule
from matsciml.common.registry import registry
from matsciml.models import NewModel


# fixture for some nominal set of hyperparameters that can be used
# across datasets
@pytest.fixture
def model_fixture() -> NewModel:
    model = NewModel(...)
    return model


# here we filter out datasets from the registry that don't make sense
ignore_dset = ["Multi", "M3G", "PyG", "Cdvae"]
filtered_list = list(
    filter(
        lambda x: all([target_str not in x for target_str in ignore_dset]),
        registry.__entries__["datasets"].keys(),
    ),
)

@pytest.mark.parametrize(
    "dset_class_name",
    filtered_list,
)
def test_model_forward_nograd(dset_class_name: str, model_fixture: NewModel):
    transforms = [
        PeriodicPropertiesTransform(cutoff_radius=6.0),
        PointCloudToGraphTransform("pyg"),
    ]
    dm = MatSciMLDataModule.from_devset(
        dset_class_name,
        batch_size=8,
        dset_kwargs={"transforms": transforms},
    )
    # dummy initialization
    dm.setup("fit")
    loader = dm.train_dataloader()
    batch = next(iter(loader))
    # run the model without gradient tracking
    with torch.no_grad():
        embeddings = model_fixture(batch)
    # returns embeddings, and runs numerical checks
    for z in [embeddings.system_embedding, embeddings.point_embedding]:
        assert torch.isreal(z).all()
        assert ~torch.isnan(z).all()  # check there are no NaNs
        assert torch.isfinite(z).all()
        assert torch.all(torch.abs(z) <= 1000)  # ensure reasonable values
```

For those unfamiliar with testing frameworks, `@pytest.fixture` instantiates the
architecture and allows re-use across tests whereas `@pytest.mark.paramterize`
will automatically generate new tests based on inputs. In this case, `filtered_list`
is created by looking at all of the datasets registered in the `registry`,
and is used to parametrize `test_model_forward_nograd` so that it is run with each
dataset.

The important thing to note is that not every test needs to pass, but having
information on which datasets work and which do not is extremely helpful for
maintainers to determine when and how to merge your new model into Open MatSciML.
On the other hand, it is also informative for you (and other users) to know,
for example, which datasets need to be normalized.

The recommendation is to nest these tests under their respective frameworks (e.g. `matsciml/models/pyg/tests`).
As you are developing your model, you can then run `pytest -vv test_<model>.py` to ensure
things are working as intended!

[egnn-test]: ./pyg/tests/test_egnn.py
