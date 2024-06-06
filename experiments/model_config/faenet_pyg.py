from matsciml.models import FAENet

from matsciml.datasets.transforms import (
    FrameAveraging,
    PeriodicPropertiesTransform,
    PointCloudToGraphTransform,
)

faenet_pyg = {
    "encoder_class": FAENet,
    "encoder_kwargs": {
        "act": "silu",
        "cutoff": 6.0,
        "average_frame_embeddings": False,
        "pred_as_dict": False,
        "hidden_dim": 128,
        "out_dim": 128,
        "tag_hidden_channels": 0,
    },
    "output_kwargs": {"lazy": False, "input_dim": 128, "hidden_dim": 128},
    "transforms": [
        PeriodicPropertiesTransform(cutoff_radius=6.5, adaptive_cutoff=True),
        PointCloudToGraphTransform(
            "pyg",
            cutoff_dist=20.0,
            node_keys=["pos", "atomic_numbers"],
        ),
        FrameAveraging(frame_averaging="3D", fa_method="stochastic"),
    ],
}
