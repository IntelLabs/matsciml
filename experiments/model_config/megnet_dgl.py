from matsciml.models import MEGNet
from matsciml.datasets.transforms import (
    DistancesTransform,
    GraphVariablesTransform,
    PeriodicPropertiesTransform,
    PointCloudToGraphTransform,
)

megnet_dgl = {
    "encoder_class": MEGNet,
    "encoder_kwargs": {
        "edge_feat_dim": 2,
        "node_feat_dim": 128,
        "graph_feat_dim": 9,
        "num_blocks": 4,
        "hiddens": [256, 256, 128],
        "conv_hiddens": [128, 128, 128],
        "s2s_num_layers": 5,
        "s2s_num_iters": 4,
        "output_hiddens": [64, 64],
        "is_classification": False,
        "encoder_only": True,
    },
    "output_kwargs": {"lazy": False, "input_dim": 640, "hidden_dim": 640},
    "transforms": [
        PeriodicPropertiesTransform(cutoff_radius=6.5, adaptive_cutoff=True),
        PointCloudToGraphTransform(
            "dgl",
            cutoff_dist=20.0,
            node_keys=["pos", "atomic_numbers"],
        ),
        DistancesTransform(),
        GraphVariablesTransform(),
    ],
}
