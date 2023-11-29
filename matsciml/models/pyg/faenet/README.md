# Frame Averaging Equivariant Network (FAENet)
## Implementation details
A high level description of the model: with frame averaging, the forward pass (`_forward`) iterates
through each canonical frame and passes it through the architecture. The core difference between this
implementation and the original is the fact that `_forward` emits an `Embeddings` data structure
consistent with the rest of the Open MatSciML Toolkit pipeline: the individual task output heads
are responsible for regression/classification, and so this FAENet _only encodes_. The `output_dim`
argument was conventionally used for property prediction (i.e. 1 for energy outputs), but we repurpose
it here as effectively the embedding dimension that gets passed into task output heads. For
the interest parties, `energy_forward` is more or less the core logic now (`_forward` -> `first_forward` -> `energy_forward`).

The rotations needed for equivariant forces have subsequently been moved out from `_forward`, and into
the respective tasks (`ForceRegressionTask` and `GradFreeRegressionTask`). In principle, the
usage should be the same as other graph-based models in `matsciml`.

The last difference to this implementation of FAENet is the additional hyperparameter,
`average_frame_embeddings` that gets passed into the `__init__` method of `FAENet`. If
set to `True`, the `_forward` method emits a single embedding for each graph/node where
we average the embeddings over frames (i.e. frame-averaged embedding).
