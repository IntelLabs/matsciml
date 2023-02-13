
from typing import Union, Optional, List, Callable, Dict
from pathlib import Path

import numpy as np
import torch
from dgl import DGLGraph
from ocpmodels.datasets.base import DGLDataset


class OE62Dataset(DGLDataset):
    def __init__(self, lmdb_root_path: Union[str, Path], cutoff_dist: float = 5., transforms: Optional[List[Callable]] = None) -> None:
        """
        Constructs an instance of `OE62Dataset`.

        This class inherits from `DGLDataset`, and the only real modifcation
        is inclusion of the `cutoff_dist` optional argument, which controls
        the distance cut off for defining an edge between atoms. This will
        then affect edge creation in the resulting molecular graph.

        Parameters
        ----------
        lmdb_root_path : Union[str, Path]
            Path to the LMDB folder, should contain *.lmdb files. In the
            case of OE62, precisely one `oe62.lmdb` file is expected as
            the dataset is small.
        cutoff_dist : float, by default 5.
            Distance between atoms to define the fictitious concept of 
            a "bond". This is used to calculate a distance matrix mask.
        transforms : Optional[List[Callable]], by default None
            List of composable transform operations, which are applied
            sequentially in the order of the list.
        """
        super().__init__(lmdb_root_path, transforms)
        self.cutoff_dist = cutoff_dist

    def data_from_key(self, lmdb_index: int, subindex: int) -> Dict[str, Union[torch.Tensor, DGLGraph]]:
        """
        Homogenize data out of the OE62 dataset, ready for batching.

        Nominally, the key names used are chosen to match the OCP data
        keys as closely as possible, just to make things consistent and
        easy to remember.

        Parameters
        ----------
        lmdb_index : int
            Index of the LMDB file; ostensibly this should just be one
            for all of OE62, since there is only one LMDB file.
        subindex
            The actual index to the sample row in the LMDB file.

        Returns
        -------
        Dict[str, Union[torch.Tensor, DGLGraph]]
            Dictionary containing a molecular graph containing atomic
            numbers and coordinates for node properties, interatomic
            distances as edge properties, and a `bandgap` regression
            target.
        """
        data = super().data_from_key(lmdb_index, subindex)
        output_data = {}
        # now we construct DGLGraphs from the loaded data
        dist_mat : np.ndarray = data.get("distance_matrix")
        lower_tri = np.tril(dist_mat)
        # mask out self loops and atoms that are too far away
        mask = (0. < lower_tri) * (lower_tri < self.cutoff_dist) 
        adj_list = np.argwhere(mask).tolist()    # DGLGraph only takes lists
        graph = DGLGraph(adj_list)
        # get coordinates, typecast to single precision
        graph.ndata["pos"] = torch.from_numpy(data.get("pos")).float()
        graph.ndata["atomic_numbers"] = torch.FloatTensor(data.get("atomic_numbers"))
        # artificially create substrate "tags" to align with OCP
        graph.ndata["tags"] = torch.ones_like(graph.ndata["atomic_numbers"]) * 2
        # get the interatomic distances as well, typecast to single precision
        graph.edata["r"] = torch.from_numpy(dist_mat[mask]).float()
        output_data["graph"] = graph
        # TODO add more targets
        output_data["bandgap"] = data.get("homo-lumo_gap")
        return output_data

