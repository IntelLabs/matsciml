from typing import List, Dict, Any, Union, Tuple
import torch
from torch.nn.utils.rnn import pad_sequence


def concatenate_keys(
    batch: List[Dict[str, Any]], pad_keys: List[str] = []
) -> Dict[str, Union[Dict[str, torch.Tensor], torch.Tensor]]:
    """
    Function for concatenating data along keys within a dictionary.

    Acts as a generic concatenation function, which can also be recursively
    applied to subdictionaries. The result is a dictionary with the same
    structure as each sample within a batch, with the exception of
    `target_keys` and `targets`, which are left blank for this dataset.

    Parameters
    ----------
    batch : List[Dict[str, Any]]
        List of samples to concatenate
    pad_keys : List[str]
        List of keys that are singled out to apply `pad_sequence` to.
        This is used for atom-centered point clouds, where the number
        of centers may not be the same between samples.

    Returns
    -------
    Dict[str, Union[Dict[str, torch.Tensor], torch.Tensor]]
        Concatenated data, following the same structure as each sample
        within `batch`.
    """
    sample = batch[0]
    batched_data = {}
    for key, value in sample.items():
        if key not in ["targets", "target_keys"]:
            if isinstance(value, dict):
                # apply function recursively on dictionaries
                result = concatenate_keys([s[key] for s in batch])
            else:
                elements = [s[key] for s in batch]
                if isinstance(value, torch.Tensor):
                    # for tensors that need to be padded
                    if key in pad_keys:
                        result = pad_sequence(elements, batch_first=True)
                    else:
                        result = torch.vstack(elements)
                # for scalar values (typically labels) pack them
                elif isinstance(value, (float, int)):
                    result = torch.tensor(elements)
                # for everything else, just return a list
                else:
                    result = elements
            batched_data[key] = result
    for key in ["targets", "target_keys"]:
        if key in sample:
            batched_data[key] = sample[key]
    return batched_data


def pad_point_cloud(data: List[torch.Tensor], max_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pads a point cloud to the maximum size within a batch.

    All this does is just initialize two tensors with an added batch dimension,
    with the number of centers/neighbors padded to the maximum point cloud
    size within a batch. This assumes "symmetric" point clouds, i.e. where
    the number of atom centers is the same as the number of neighbors.

    Parameters
    ----------
    data : List[torch.Tensor]
        List of point cloud data to batch
    max_size : int
        Number of particles per point cloud

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        Returns the padded data, along with a mask
    """
    batch_size = len(data)
    # get the feature dimension
    feat_dim = data[0].size(-1)
    result = torch.zeros((batch_size, max_size, max_size, feat_dim), dtype=data[0].dtype)
    mask = torch.zeros((batch_size, max_size, max_size), dtype=torch.bool)
    for index, entry in enumerate(data):
        assert entry.size(0) == entry.size(1), f"Point cloud padding assumes the same number of centers and neighbors."
        num_particles = entry.size(0)
        # copy over data
        result[index, :num_particles, :num_particles, :] = entry
        # this indicates which elements correspond to unpadded stuff
        mask[index, :num_particles, :num_particles] = True
    return (result, mask)


def point_cloud_featurization(
    src_types: torch.Tensor, dst_types: torch.Tensor, max_types: int = 100
) -> torch.Tensor:
    """
    Featurizes an atom-centered point cloud, given source and destination node types.

    Takes integer encodings of node types for both source (atom-centers) and destination (neighborhood),
    and converts them into one-hot encodings that take +/- combinations.

    Parameters
    ----------
    src_types : torch.Tensor
        1D tensor containing node types for centers
    dst_types : torch.Tensor
        1D tensor containing node types for neighbors
    max_types : int
        Maximum value for node types, default 100

    Returns
    -------
    torch.Tensor
        Feature tensor, with a shape of [num_src, num_dst, 2 x max_types]
    """
    eye = torch.eye(max_types)
    src_onehot = eye[src_types][:, None]
    dst_onehot = eye[dst_types][None, :]
    plus, minus = src_onehot + dst_onehot, src_onehot - dst_onehot
    feat_tensor = torch.concat([plus, minus], axis=-1)
    return feat_tensor
