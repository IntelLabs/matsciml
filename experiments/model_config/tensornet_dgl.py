from matsciml.datasets.utils import element_types
from matsciml.models import TensorNet

from matsciml.datasets.transforms import (
    PeriodicPropertiesTransform,
    PointCloudToGraphTransform,
)

tensornet_dgl = {
    "encoder_class": TensorNet,
    "encoder_kwargs": {
        "element_types": element_types(),
        "num_rbf": 32,
        "max_n": 3,
        "max_l": 3,
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
