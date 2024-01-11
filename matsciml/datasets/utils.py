from __future__ import annotations

import pickle
from collections.abc import Generator
from functools import lru_cache, partial
from os import makedirs
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import lmdb
import torch
from joblib import Parallel, delayed
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from matsciml.common import package_registry
from matsciml.common.types import BatchDict, DataDict, GraphTypes

if package_registry["dgl"]:
    import dgl

if package_registry["pyg"]:
    import torch_geometric
    from torch_geometric.data import Batch as PyGBatch
    from torch_geometric.data import Data as PyGGraph


def concatenate_keys(
    batch: list[DataDict],
    pad_keys: list[str] = [],
    unpacked_keys: list[str] = [],
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
            if isinstance(value, str):
                result = value
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
                                        [max(t.shape[:-1]) for t in elements],
                                    )
                                result, mask = pad_point_cloud(
                                    elements,
                                    max_size=max_size,
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
                                value,
                                dgl.DGLGraph,
                            ):
                                result = dgl.batch(elements)
                            elif package_registry["pyg"] and isinstance(
                                value,
                                PyGGraph,
                            ):
                                result = PyGBatch.from_data_list(elements)
                            else:
                                raise ValueError(
                                    f"Graph type unsupported: {type(value)}",
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
    data: list[torch.Tensor],
    max_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
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
    src_types: torch.Tensor,
    dst_types: torch.Tensor,
    max_types: int = 100,
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


def connect_db_read(lmdb_path: str | Path, **kwargs) -> lmdb.Environment:
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


def connect_lmdb_write(
    lmdb_target_file: str | Path,
    **kwargs,
) -> lmdb.Environment:
    """
    Open an LMDB environment for writing.

    This function will enforce the ``.lmdb`` file extension if it
    is not already present in the filepath. Kwargs are passed
    into ``lmdb.open``

    Parameters
    ----------
    lmdb_target_file : Union[str, Path]
        Target path to open an LMDB file

    Returns
    -------
    lmdb.Environment
        Open LMDB environment for writing
    """
    kwargs.setdefault("map_size", 1099511627776 * 2)
    kwargs.setdefault("meminit", False)
    kwargs.setdefault("subdir", False)
    kwargs.setdefault("map_async", True)
    if isinstance(lmdb_target_file, str):
        lmdb_target_file = Path(lmdb_target_file)
    # make sure we append the file extension
    lmdb_target_file = lmdb_target_file.with_suffix(".lmdb")
    # convert to string to be passed into lmdb.open
    lmdb_target_file = str(lmdb_target_file)
    output_env = lmdb.open(lmdb_target_file, **kwargs)
    return output_env


def get_lmdb_keys(
    env: lmdb.Environment,
    ignore_keys: list[str] | None = None,
    _lambda: Callable | None = None,
) -> list[str]:
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
            f"Both `ignore_keys` and `_lambda` were passed; arguments are mutually exclusive.",
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
    get_lmdb_keys,
    _lambda=lambda x: x.isnumeric(),
    ignore_keys=None,
)


def get_lmdb_data_length(lmdb_path: str | Path) -> int:
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
    db_index: int,
    data_index: int,
    envs: list[lmdb.Environment],
) -> dict[str, Any]:
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
    except IndexError as error:
        error(
            f"Tried to retrieve LMDB file {db_index}, but only {len(envs)} are loaded.",
        )
    with env.begin() as txn:
        data = pickle.loads(txn.get(f"{data_index}".encode("ascii")))
        if not data:
            raise ValueError(
                f"Data sample at index {data_index} for file {env.path()} missing.",
            )
    return data


def get_lmdb_metadata(target_lmdb: lmdb.Environment) -> dict[str, Any] | None:
    """
    Load in metadata associated with a specific LMDB file.

    Returns the dictionary if it's present, otherwise returns ``None``.

    Parameters
    ----------
    target_lmdb : lmdb.Environment
        Target LMDB file to inspect

    Returns
    -------
    Union[Dict[str, Any], None]
        None if no metadata present, otherwise the metadata dictionary.
    """
    with target_lmdb.begin() as txn:
        metadata = txn.get(b"metadata")
    if metadata:
        return pickle.loads(metadata)
    else:
        return None


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


def parallel_lmdb_write(
    target_dir: str | Path,
    data: list[Any],
    num_procs: int,
    metadata: dict[str, Any] | None = None,
) -> None:
    r"""
    Writes a set of data out to LMDB file, parallelized over ``num_procs`` workers.

    The user specifies a folder that will comprise all of the individual LMDB
    files, where each file consists of ``data`` split up into ``num_procs`` number
    of data chunks.

    Parameters
    ----------
    target_dir : Union[str, Path]
        Directory containing LMDB files to target. If this doesn't exist
        it will be created.
    data : List[Any]
        Data to save to LMDB file(s).
    num_procs : int
        Number of processes to split the writing task to
    metadata : Optional[Dict[str, Any]], optional
        Metadata to write to disk. Corresponds to a dictionary with
        key/value pairs that denote extra information the user wishes
        to save with the dataset.
    """
    if isinstance(target_dir, str):
        target_dir = Path(target_dir)
    # make the LMDB directory
    makedirs(target_dir, exist_ok=True)
    assert target_dir.is_dir(), f"Target to write LMDB data to is not a directory."

    def write_chunk(
        chunk: list[Any],
        target_dir: Path,
        index: int,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        r"""
        Write a chunk of data to a correspond LMDB environment.

        Parameters
        ----------
        chunk : List[Any]
            List of data to write
        target_dir : Path
            Directory containing LMDB files to target
        index : int
            Index of this processor, used to append to the LMDB filename
        preprocessed : bool, optional
            Indicates if this data is preprocessed, by default False
        """
        formatted_index = str(index).zfill(4)
        lmdb_target = target_dir.joinpath(f"data.{formatted_index}.lmdb")
        lmdb_env = connect_lmdb_write(lmdb_target)
        for subindex, _data in enumerate(
            tqdm(
                chunk,
                position=index,
                total=len(chunk),
                desc=f"Writing LMDB data to file {lmdb_env.path()}",
            ),
        ):
            write_lmdb_data(subindex, _data, lmdb_env)
        if metadata:
            write_lmdb_data("metadata", metadata, lmdb_env)

    def divide_data_chunks(
        all_data: list[Any],
        num_chunks: int,
    ) -> Generator[list[Any], None, None]:
        r"""
        Split data into a specified number of chunks.

        The number of samples per chunk should be roughly equal
        to the extent it is possible.

        Parameters
        ----------
        all_data : List[Any]
            List of data to divvy up into chunks
        num_chunks : int
            Number of chunks to split the data into

        Yields
        ------
        Generator[List[Any]]
            Generator that will iteratively emit chunks of data
        """
        for i in range(num_chunks):
            yield all_data[i::num_chunks]

    chunks = list(divide_data_chunks(data, num_procs))
    lengths = [len(chunk) for chunk in chunks]
    lmdb_indices = list(range(num_procs))
    assert all(
        [length != 0 for length in lengths],
    ), f"Too many processes specified and not enough data to split over multiple LMDB files. Decrease `num_procs!`"
    p = Parallel(num_procs)(
        delayed(write_chunk)(chunk, target_dir, index, metadata)
        for chunk, index in zip(chunks, lmdb_indices)
    )


def retrieve_pointcloud_node_types(pc_feats: torch.Tensor) -> tuple[torch.Tensor]:
    r"""
    Attempt to reproduce the original node types from the
    molecule-centered point cloud featurization.

    Essentially just sums along the ``src`` and ``dst`` dimensions
    for a single point cloud sample, and returns the index which
    was the largest.

    Parameters
    ----------
    pc_feats : torch.Tensor
        3D tensor with shape ``[src_nodes, dst_nodes, atom_types]``
        to retrieve the node types from.

    Returns
    -------
    Tuple[torch.Tensor]
        Pair of tensors with the original atomic numbers.
    """
    assert (
        pc_feats.ndim == 3
    ), f"Expected individual samples of point clouds, not batched."
    src_types = pc_feats.sum(dim=1).argmax(-1)
    dst_types = pc_feats.sum(dim=0).argmax(-1)
    return (src_types, dst_types)


@lru_cache(maxsize=1)
def atomic_number_map() -> dict[str, int]:
    """List of element symbols and their atomic numbers.

    Returns:
        Dict[str, int]: _description_
    """
    # fmt: off
    an_map = {
        'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8,
        'F': 9, 'Ne': 10, 'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15,
        'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22,
        'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29,
        'Zn': 30, 'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36,
        'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40, 'Nb': 41, 'Mo': 42, 'Tc': 43,
        'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50,
        'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55, 'Ba': 56, 'La': 57,
        'Ce': 58, 'Pr': 59, 'Nd': 60, 'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64,
        'Tb': 65, 'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70, 'Lu': 71,
        'Hf': 72, 'Ta': 73, 'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78,
        'Au': 79, 'Hg': 80, 'Tl': 81, 'Pb': 82, 'Bi': 83, 'Po': 84, 'At': 85,
        'Rn': 86, 'Fr': 87, 'Ra': 88, 'Ac': 89, 'Th': 90, 'Pa': 91, 'U': 92,
        'Np': 93, 'Pu': 94, 'Am': 95, 'Cm': 96, 'Bk': 97, 'Cf': 98, 'Es': 99,
        'Fm': 100, 'Md': 101, 'No': 102, 'Lr': 103, 'Rf': 104, 'Db': 105,
        'Sg': 106, 'Bh': 107, 'Hs': 108, 'Mt': 109, 'Ds': 110, 'Rg': 111,
        'Cn': 112, 'Nh': 113, 'Fl': 114, 'Mc': 115, 'Lv': 116, 'Ts': 117,
        'Og': 118,
    }
    # fmt: on
    return an_map


@lru_cache(1)
def element_types():
    return list(atomic_number_map().keys())
