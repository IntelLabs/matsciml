###########################################################################################
# Implementation of MACE models and other models based E(3)-Equivariant MPNNs
# (https://github.com/ACEsuit/mace)
# Original Authors: Ilyes Batatia, Gregor Simm
# Integrated into matsciml by Vaibhav Bihani, Sajid Mannan
# This program is distributed under the MIT License
###########################################################################################


from __future__ import annotations

from matsciml.models.pyg.mace.tools.cg import U_matrix_real
from matsciml.models.pyg.mace.tools.checkpoint import (
    CheckpointHandler,
    CheckpointIO,
    CheckpointState,
)
from matsciml.models.pyg.mace.tools.torch_tools import (
    TensorDict,
    cartesian_to_spherical,
    count_parameters,
    init_device,
    init_wandb,
    set_default_dtype,
    set_seeds,
    spherical_to_cartesian,
    to_numpy,
    to_one_hot,
    voigt_to_matrix,
)
from matsciml.models.pyg.mace.tools.utils import (
    AtomicNumberTable,
    MetricsLogger,
    atomic_numbers_to_indices,
    compute_c,
    compute_mae,
    compute_q95,
    compute_rel_mae,
    compute_rel_rmse,
    compute_rmse,
    get_atomic_number_table_from_zs,
    get_optimizer,
    get_tag,
    setup_logger,
)

__all__ = [
    "TensorDict",
    "AtomicNumberTable",
    "atomic_numbers_to_indices",
    "to_numpy",
    "to_one_hot",
    "build_default_arg_parser",
    "set_seeds",
    "init_device",
    "setup_logger",
    "get_tag",
    "count_parameters",
    "get_optimizer",
    "MetricsLogger",
    "get_atomic_number_table_from_zs",
    "CheckpointHandler",
    "CheckpointIO",
    "CheckpointState",
    "set_default_dtype",
    "compute_mae",
    "compute_rel_mae",
    "compute_rmse",
    "compute_rel_rmse",
    "compute_q95",
    "compute_c",
    "U_matrix_real",
    "spherical_to_cartesian",
    "cartesian_to_spherical",
    "voigt_to_matrix",
    "init_wandb",
]
