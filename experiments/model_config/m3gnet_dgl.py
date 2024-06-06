from matsciml.datasets.utils import element_types
from matsciml.models import M3GNet

from matsciml.datasets.transforms import (
    MGLDataTransform,
    PeriodicPropertiesTransform,
    PointCloudToGraphTransform,
)

m3gnet_dgl = {
    "encoder_class": M3GNet,
    "encoder_kwargs": {
        "element_types": element_types(),
        "return_all_layer_output": True,
    },
    "output_kwargs": {"lazy": False, "input_dim": 64, "hidden_dim": 64},
    "transforms": [
        PeriodicPropertiesTransform(cutoff_radius=6.5, adaptive_cutoff=True),
        PointCloudToGraphTransform(
            "dgl",
            cutoff_dist=20.0,
            node_keys=["pos", "atomic_numbers"],
        ),
        MGLDataTransform(),
    ],
}
