from torch.nn import LayerNorm, SiLU

from matsciml.models import PLEGNNBackbone

from matsciml.datasets.transforms import (
    PeriodicPropertiesTransform,
    PointCloudToGraphTransform,
)

egnn_dgl = {
    "encoder_class": PLEGNNBackbone,
    "encoder_kwargs": {
        "embed_in_dim": 1,
        "embed_hidden_dim": 32,
        "embed_out_dim": 128,
        "embed_depth": 5,
        "embed_feat_dims": [128, 128, 128],
        "embed_message_dims": [128, 128, 128],
        "embed_position_dims": [64, 64],
        "embed_edge_attributes_dim": 0,
        "embed_activation": "relu",
        "embed_residual": True,
        "embed_normalize": True,
        "embed_tanh": True,
        "embed_activate_last": False,
        "embed_k_linears": 1,
        "embed_use_attention": False,
        "embed_attention_norm": "sigmoid",
        "readout": "sum",
        "node_projection_depth": 3,
        "node_projection_hidden_dim": 128,
        "node_projection_activation": "relu",
        "prediction_out_dim": 1,
        "prediction_depth": 3,
        "prediction_hidden_dim": 128,
        "prediction_activation": "relu",
    },
    "output_kwargs": {
        "norm": LayerNorm(128),
        "hidden_dim": 128,
        "activation": SiLU,
        "lazy": False,
        "input_dim": 128,
    },
    "transforms": [
        PeriodicPropertiesTransform(cutoff_radius=6.5, adaptive_cutoff=True),
        PointCloudToGraphTransform(
            "dgl",
            cutoff_dist=20.0,
            node_keys=["pos", "atomic_numbers"],
        ),
    ],
}
