from __future__ import annotations

from inspect import signature
from logging import getLogger
from typing import Any

from e3nn.o3 import Irreps
from mace.modules import MACE

from matsciml.models.base import AbstractPyGModel
from matsciml.common.registry import registry


__mace_signature = signature(MACE)
__mace_parameters = __mace_signature.parameters


logger = getLogger(__file__)


@registry.register_model("MACEWrapper")
class MACEWrapper(AbstractPyGModel):
    def __init__(
        self,
        atom_embedding_dim: int,
        num_atom_embedding: int = 100,
        embedding_kwargs: Any = None,
        encoder_only: bool = True,
        **mace_kwargs,
    ) -> None:
        if embedding_kwargs is not None:
            logger.warning("`embedding_kwargs` is not used for MACE models.")
        super().__init__(atom_embedding_dim, num_atom_embedding, {}, encoder_only)
        for key in mace_kwargs:
            assert (
                key in __mace_parameters
            ), f"{key} was passed as a MACE kwarg but does not match expected arguments."
        # remove the embedding table, as MACE uses e3nn layers
        del self.atom_embedding
        if "num_elements" in mace_kwargs:
            raise KeyError(
                "Please use `num_atom_embedding` instead of passing `num_elements`."
            )
        if "hidden_irreps" in mace_kwargs:
            raise KeyError(
                "Please use `atom_embedding_dim` instead of passing `hidden_irreps`."
            )
        atom_embedding_dim = Irreps(f"{atom_embedding_dim}x0e")
        self.encoder = ...
