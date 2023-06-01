from typing import List, Dict, Any, Union
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
