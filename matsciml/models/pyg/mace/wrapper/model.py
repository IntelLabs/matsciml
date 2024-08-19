from __future__ import annotations

from logging import getLogger
from typing import Any, Callable, Type
from functools import cache

import torch
from e3nn.o3 import Irreps
from mace.modules import MACE
from torch_geometric.nn import pool
from mendeleev import element

from matsciml.models.base import AbstractPyGModel
from matsciml.common.types import BatchDict, DataDict, AbstractGraph, Embeddings
from matsciml.common.registry import registry
from matsciml.common.inspection import get_model_required_args, get_model_all_args


logger = getLogger(__file__)

__all__ = ["MACEWrapper"]


def free_ion_energy_table(num_elements: int = 100) -> torch.Tensor:
    """
    Generates a default table of atomic energies as the total
    stripped ion energy. Keep in mind that the sign is negated.

    Parameters
    ----------
    num_elements : int
        Number of atoms to include in the tensor. This must
        match the `num_atom_embedding` argument of ``MACE``,
        otherwise will encounter a matmul error. Defaults to
        100, which is the default value for ``MACEWrapper``.

    Returns
    -------
    torch.Tensor
        Tensor containing the energies of fully stripped ions
        in double precision.
    """
    ele = [element(i) for i in range(1, num_elements + 1)]
    ion_energies = [-sum(e.ionenergies.values()) for e in ele]
    return torch.Tensor(ion_energies).double()


@registry.register_model("MACEWrapper")
class MACEWrapper(AbstractPyGModel):
    def __init__(
        self,
        atom_embedding_dim: int,
        mace_module: Type[MACE] = MACE,
        num_atom_embedding: int = 100,
        embedding_kwargs: Any = None,
        encoder_only: bool = True,
        readout_method: str | Callable = "add",
        **mace_kwargs,
    ) -> None:
        if embedding_kwargs is not None:
            logger.warning("`embedding_kwargs` is not used for MACE models.")
        super().__init__(atom_embedding_dim, num_atom_embedding, {}, encoder_only)
        # dynamically check to check which arguments are needed by MACE
        __mace_required_args = get_model_required_args(MACE)
        __mace_all_args = get_model_all_args(MACE)

        __mace_submodule_required_args = get_model_required_args(mace_module)
        __mace_submodule_all_args = get_model_all_args(mace_module)
        if "kwargs" in __mace_submodule_required_args:
            __mace_submodule_required_args.remove("kwargs")
        if "kwargs" in __mace_submodule_all_args:
            __mace_submodule_all_args.remove("kwargs")
        for key in mace_kwargs:
            assert (
                key in __mace_all_args + __mace_submodule_all_args
            ), f"{key} was passed as a MACE kwarg but does not match expected arguments."
        # remove the embedding table, as MACE uses e3nn layers
        del self.atom_embedding
        if "num_elements" in mace_kwargs:
            raise KeyError(
                "Please use `num_atom_embedding` instead of passing `num_elements`."
            )
        hidden_irreps = mace_kwargs.get(
            "hidden_irreps", Irreps(f"{atom_embedding_dim}x0e")
        )
        # pack stuff into the mace kwargs
        mace_kwargs["num_elements"] = num_atom_embedding
        mace_kwargs["hidden_irreps"] = hidden_irreps
        mace_kwargs["atomic_numbers"] = list(range(1, num_atom_embedding + 1))
        if "atomic_energies" not in mace_kwargs:
            logger.warning(
                "No ``atomic_energies`` provided, defaulting to total ionization energy."
            )
            mace_kwargs["atomic_energies"] = free_ion_energy_table(num_atom_embedding)
        # check to make sure all that's required is
        for key in __mace_required_args + __mace_submodule_required_args:
            if key not in mace_kwargs:
                raise KeyError(
                    f"{key} is required by MACE, but was not found in kwargs."
                )
        self.encoder = mace_module(**mace_kwargs)
        # if a string is passed, grab the PyG builtins
        if isinstance(readout_method, str):
            readout_type = getattr(pool, f"global_{readout_method}_pool", None)
            if not readout_type:
                possible_methods = list(filter(lambda x: "global" in x, dir(pool)))
                raise NotImplementedError(
                    f"{readout_method} is not a valid function in PyG pooling."
                    f" Supported methods are: {possible_methods}"
                )
            readout_method = readout_type
        self.readout = readout_method
        self.save_hyperparameters()

    @property
    @cache
    def _atom_eye(self) -> torch.Tensor:
        return torch.eye(
            self.hparams.num_atom_embedding, device=self.device, dtype=self.dtype
        )

    def atomic_numbers_to_one_hot(self, atomic_numbers: torch.Tensor) -> torch.Tensor:
        """
        Convert discrete atomic numbers into one-hot vectors based
        on some maximum number of elements possible.

        Parameters
        ----------
        atomic_numbers : torch.Tensor
            1D tensor of integers corresponding to atomic numbers.

        Returns
        -------
        torch.Tensor
            2D tensor of one-hot vectors for each node.
        """
        return self._atom_eye[atomic_numbers.long()]

    def read_batch(self, batch: BatchDict) -> DataDict:
        data = {}
        # expect a PyG graph already
        graph = batch["graph"]
        atomic_numbers = graph.atomic_numbers
        one_hot_atoms = self.atomic_numbers_to_one_hot(atomic_numbers)
        # check to make sure we have unit cell shifts
        for key in ["cell", "offsets"]:
            if key not in batch:
                raise KeyError(
                    f"Expected periodic property {key} to be in batch."
                    " Please include ``PeriodicPropertiesTransform``."
                )
        # hacky way of letting MACE work for single graph
        if "batch" not in graph:
            from torch_geometric.data import Batch
            graph = Batch.from_data_list([graph])
        assert hasattr(graph, "ptr"), "Graph is missing the `ptr` attribute!"
        # the name of these keys matches up with our `_forward`, and
        # later get remapped to MACE ones
        data.update(
            {
                "graph": graph,
                "pos": graph.pos,
                "node_feats": one_hot_atoms,
                "cell": batch["cell"],
                "shifts": batch["offsets"],
            }
        )
        return data

    def _forward(
        self,
        graph: AbstractGraph,
        node_feats: torch.Tensor,
        pos: torch.Tensor,
        **kwargs,
    ) -> Embeddings:
        """
        Takes arguments in the standardized format, and passes them into MACE
        with some redundant mapping.

        Parameters
        ----------
        graph : AbstractGraph
            Graph structure containing node and graph properties

        node_feats : torch.Tensor
            Tensor containing one-hot node features, shape ``[num_nodes, num_elements]``

        pos : torch.Tensor
            2D tensor containing node positions, shape ``[num_nodes, 3]``

        Returns
        -------
        Embeddings
            MatSciML ``Embeddings`` structure
        """
        # repack data into MACE format
        mace_data = {
            "positions": pos,
            "node_attrs": node_feats,
            "ptr": graph.ptr,
            "cell": kwargs["cell"],
            "shifts": kwargs["shifts"],
            "batch": graph.batch,
            "edge_index": graph.edge_index,
        }
        outputs = self.encoder(
            mace_data,
            training=self.training,
            compute_force=False,
            compute_virials=False,
            compute_stress=False,
            compute_displacement=False,
        )
        node_embeddings = outputs["node_feats"]
        graph_embeddings = self.readout(node_embeddings, graph.batch)
        return Embeddings(graph_embeddings, node_embeddings)
