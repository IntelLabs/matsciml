from typing import List, Dict, Any, Union, Tuple, Optional, Callable
from pathlib import Path
import torch
from torch.nn.utils.rnn import pad_sequence
import lmdb
import pickle
from tqdm import tqdm

from ocpmodels.common.types import DataDict, BatchDict, GraphTypes
from ocpmodels.common import package_registry

if package_registry["dgl"]:
    import dgl

if package_registry["pyg"]:
    import torch_geometric
    from torch_geometric.data import Data as PyGGraph
    from torch_geometric.data import Batch as PyGBatch


def concatenate_keys(
    batch: List[DataDict], pad_keys: List[str] = [], unpacked_keys: List[str] = []
) -> BatchDict:
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
        if key not in ["target_types", "target_keys"]:
            if isinstance(value, dict):
                # apply function recursively on dictionaries
                result = concatenate_keys([s[key] for s in batch])
            else:
                elements = [s[key] for s in batch]
                # provides an escape hatch; sometimes we don't want to stack
                # tensors together and instead just leave them as a list
                if key not in unpacked_keys:
                    try:
                        if isinstance(value, torch.Tensor):
                            # for tensors that need to be padded
                            if key in pad_keys:
                                # for 1D tensors like atomic numbers, we need to know the
                                # maximum number of nodes
                                if value.ndim == 1:
                                    max_size = max([len(t) for t in elements])
                                else:
                                    # for other tensors, pad
                                    max_size = max(
                                        [max(t.shape[:-1]) for t in elements]
                                    )
                                result, mask = pad_point_cloud(
                                    elements, max_size=max_size
                                )
                                batched_data["mask"] = mask
                            else:
                                result = torch.vstack(elements)
                        # for scalar values (typically labels) pack them, add a dimension
                        # to match model predictions, and type cast to float
                        elif isinstance(value, (float, int)):
                            result = torch.tensor(elements).unsqueeze(-1).float()
                        # for graph types, descend into framework specific method
                        elif isinstance(value, GraphTypes):
                            if package_registry["dgl"] and isinstance(
                                value, dgl.DGLGraph
                            ):
                                result = dgl.batch(elements)
                            elif package_registry["pyg"] and isinstance(
                                value, PyGGraph
                            ):
                                result = PyGBatch.from_data_list(elements)
                            else:
                                raise ValueError(
                                    f"Graph type unsupported: {type(value)}"
                                )
                    except RuntimeError:
                        result = elements
                # for everything else, just return a list
                else:
                    result = elements
            batched_data[key] = result
    for key in ["target_types", "target_keys"]:
        if key in sample:
            batched_data[key] = sample[key]
    return batched_data


def pad_point_cloud(
    data: List[torch.Tensor], max_size: int
) -> Tuple[torch.Tensor, torch.Tensor]:
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
    data_dim = data[0].dim()
    # get the feature dimension
    if data_dim == 1:
        feat_dim = max_size
    else:
        feat_dim = data[0].size(-1)
    zeros_dims = [batch_size, *[max_size] * (data_dim - 1), feat_dim]
    result = torch.zeros((zeros_dims), dtype=data[0].dtype)
    mask = torch.zeros((zeros_dims[:-1]), dtype=torch.bool)

    for index, entry in enumerate(data):
        # Get all indices from entry, we we can use them to pad result. Add batch idx to the beginning.
        indices = [torch.tensor(index)] + [torch.arange(size) for size in entry.shape]
        indices = torch.meshgrid(*indices, indexing="ij")
        # Use the index_put method to pop entry into result.
        result.index_put_(tuple(indices), entry)
        mask.index_put_(indices[:-1], torch.tensor(True))

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


def connect_db_read(lmdb_path: Union[str, Path], **kwargs) -> lmdb.Environment:
    """
    Open an LMDB file for reading.

    Additional ``kwargs`` can be passed to modify the read behavior,
    however by definition the ``readonly`` kwarg will always be set to
    ``True``.

    ``kwargs`` are passed into ``lmdb.open``.

    Parameters
    ----------
    lmdb_path : Union[str, Path]
        Path to an LMDB folder structure

    Returns
    -------
    lmdb.Environment
        LMDB object
    """
    kwargs.setdefault("subdir", False)
    kwargs.setdefault("lock", False)
    kwargs.setdefault("readahead", False)
    kwargs.setdefault("meminit", False)
    kwargs.setdefault("max_readers", 1)
    # force ignore readonly overriding
    if "readonly" in kwargs:
        del kwargs["readonly"]
    if isinstance(lmdb_path, Path):
        lmdb_path = str(lmdb_path)
    env = lmdb.open(lmdb_path, readonly=True, **kwargs)
    return env


def get_lmdb_keys(
    env: lmdb.Environment,
    ignore_keys: Optional[List[str]] = None,
    _lambda: Optional[Callable] = None,
) -> List[str]:
    """
    Utility function to get keys from an LMDB file.

    Provides the ability to filter out certain keys, and will
    return a sorted list. The two modes of operation for this
    filtering action is to either provide a list of keys to ignore,
    or a ``lambda`` function that will be applied to each key.

    Parameters
    ----------
    env : lmdb.Environment
        Instance of an ``lmdb.Environment`` object.
    ignore_keys : Optional[List[str]], optional
        Optional list of keys to ignore, by default None which
        will return all keys.
    _lambda : Optional[Callback], optional
        Function used to filter the list of keys, by default None

    Returns
    -------
    List[str]
        Sorted list of filtered keys contained in the LMDB file
    """
    with env.begin() as txn:
        keys = [key.decode("utf-8") for key in txn.cursor().iternext(values=False)]
    if ignore_keys and _lambda:
        raise ValueError(
            f"Both `ignore_keys` and `_lambda` were passed; arguments are mutually exclusive."
        )
    if ignore_keys:
        _lambda = lambda x: x not in ignore_keys
    else:
        if not _lambda:
            # escape case where we basically don't filter
            _lambda = lambda x: x
    # convert to a sorted list of keys
    keys = sorted(list(filter(_lambda, keys)))
    return keys


# this provides a quick way to get only data keys from an LMDB
get_lmdb_data_keys = partial(
    get_lmdb_keys, _lambda=lambda x: x.isnumeric(), ignore_keys=None
)


def get_lmdb_data_length(lmdb_path: Union[str, Path]) -> int:
    """
    Retrieve the number of data entries within a LMDB file.

    This uses ``get_lmdb_data_keys`` to extract only numeric keys
    within the LMDB file, i.e. assumes only integer valued keys
    contain data samples.

    Parameters
    ----------
    lmdb_path : Union[str, Path]
        Path to the LMDB file.

    Returns
    -------
    int
        Number of data samples
    """
    env = connect_db_read(lmdb_path)
    # this gets the number of data keys
    keys = get_lmdb_data_keys(env)
    length = len(keys)
    return length


def get_data_from_index(
    db_index: int, data_index: int, envs: List[lmdb.Environment]
) -> Dict[str, Any]:
    """
    Given a pair of indices, retrieve a data sample.

    The indices are used to first look up which LMDB environment
    to look into, followed by the index within that file.

    Parameters
    ----------
    db_index : int
        Index for the LMDB environment within `envs`.
    data_index : int
        Index for the data sample within an LMDB environment.
    envs : List[lmdb.Environment]
        List of `lmdb.Environment` objects

    Returns
    -------
    Dict[str, Any]
        Data sample retrieved from the environments
    """
    try:
        env = envs[db_index]
    except IndexError(f"Tried to retrieve LMDB file {db_index}, but only {len(envs)} are loaded.")
    with env.begin() as txn:
        data = pickle.loads(txn.get(f"{data_index}".encode("ascii")))
        if not data:
            raise ValueError(f"Data sample at index {data_index} for file {env.path} missing.")
    return data


def write_lmdb_data(key: Any, data: Any, target_lmdb: lmdb.Environment) -> None:
    """
    Write a dictionary of data to an LMDB output.

    Uses ``pickle`` to dump data using the highest protocol available. Keys
    are first converted from any data type (i.e. integers) into a string
    with ``ascii`` encoding.

    Parameters
    ----------
    key : Any
        Key to store data to within `target_lmdb`
    data : Any
        Any picklable object to save to `target_lmdb`
    target_lmdb : lmdb.Environment
        LMDB environment to save data to
    """
    with target_lmdb.begin(write=True) as txn:
        txn.put(key=f"{key}".encode("ascii"), value=pickle.dumps(data, protocol=-1))