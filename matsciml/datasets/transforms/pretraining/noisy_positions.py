from __future__ import annotations

import torch
from dgl import DGLGraph

from matsciml.common.types import DataDict
from matsciml.datasets.transforms.base import AbstractDataTransform


__all__ = ["NoisyPositions"]


class NoisyPositions(AbstractDataTransform):
    def __init__(self, scale: float = 1e-3) -> None:
        """
        Initializes a NoisyPositions transform.

        This class generates i.i.d. Gaussian displacements to atom
        coordinates, and adds a new ``noisy_pos`` key to the data
        sample. While there is no prescribed ordering of transforms,
        if graphs are available in the sample the transform will
        act on the graph over the raw point cloud positions. If
        your pipeline involves graph creation, note that this _could_
        affect the resulting edges produced, depending on the scale of
        noise used.

        Implemented from the strategy described by Zaidi _et al._ 2023,
        https://openreview.net/pdf?id=tYIMtogyee

        Parameters
        ----------
        scale : float
            Scale used to multiply N~(0, I_3) Gaussian noise
        """
        super().__init__()
        self.scale = scale

    def __call__(self, data: DataDict) -> DataDict:
        if "graph" in data:
            graph = data["graph"]
            if isinstance(graph, DGLGraph):
                pos = graph.ndata["pos"]
            else:
                # assume it's a PyG graph, grab as attribute
                pos = graph.pos
        else:
            # otherwise it's a point cloud
            pos = data["pos"]
        noise = torch.randn_like(pos) * self.scale
        noisy_pos = pos + noise
        # write the noisy node data; same logic as before
        if "graph" in data:
            graph = data["graph"]
            if isinstance(graph, DGLGraph):
                graph.ndata["noisy_pos"] = noisy_pos
            else:
                setattr(graph, "noisy_pos", noisy_pos)
        else:
            data["noisy_pos"] = noisy_pos
        # set targets so that tasks know what to do
        data["targets"]["denoise"] = noise
        if "pretraining" in data["target_types"]:
            data["target_types"]["pretraining"].append("denoise")
        else:
            data["target_types"]["pretraining"] = ["denoise"]
        return data
