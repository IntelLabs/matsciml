###########################################################################################
# Implementation of MACE models and other models based E(3)-Equivariant MPNNs
# (https://github.com/ACEsuit/mace)
# Original Authors: Ilyes Batatia, Gregor Simm
# Integrated into matsciml by Vaibhav Bihani, Sajid Mannan
# This program is distributed under the MIT License
###########################################################################################


from __future__ import annotations

from typing import Callable

import torch

from matsciml.models.pyg.mace.modules.blocks import (
    AgnosticNonlinearInteractionBlock,
    AgnosticResidualNonlinearInteractionBlock,
    AtomicEnergiesBlock,
    EquivariantProductBasisBlock,
    InteractionBlock,
    LinearDipoleReadoutBlock,
    LinearNodeEmbeddingBlock,
    LinearReadoutBlock,
    NonLinearDipoleReadoutBlock,
    NonLinearReadoutBlock,
    RadialEmbeddingBlock,
    RealAgnosticInteractionBlock,
    RealAgnosticResidualInteractionBlock,
    ResidualElementDependentInteractionBlock,
    ScaleShiftBlock,
)
from matsciml.models.pyg.mace.modules.models import (
    MACE,
    AtomicDipolesMACE,
    BOTNet,
    EnergyDipolesMACE,
    ScaleShiftBOTNet,
    ScaleShiftMACE,
)
from matsciml.models.pyg.mace.modules.radial import BesselBasis, PolynomialCutoff
from matsciml.models.pyg.mace.modules.symmetric_contraction import SymmetricContraction
from matsciml.models.pyg.mace.modules.utils import (
    compute_avg_num_neighbors,
    compute_fixed_charge_dipole,
    compute_mean_rms_energy_forces,
    compute_mean_std_atomic_inter_energy,
    compute_rms_dipoles,
)

interaction_classes: dict[str, type[InteractionBlock]] = {
    "AgnosticNonlinearInteractionBlock": AgnosticNonlinearInteractionBlock,
    "ResidualElementDependentInteractionBlock": ResidualElementDependentInteractionBlock,
    "AgnosticResidualNonlinearInteractionBlock": AgnosticResidualNonlinearInteractionBlock,
    "RealAgnosticResidualInteractionBlock": RealAgnosticResidualInteractionBlock,
    "RealAgnosticInteractionBlock": RealAgnosticInteractionBlock,
}

scaling_classes: dict[str, Callable] = {
    "std_scaling": compute_mean_std_atomic_inter_energy,
    "rms_forces_scaling": compute_mean_rms_energy_forces,
    "rms_dipoles_scaling": compute_rms_dipoles,
}

gate_dict: dict[str, Callable | None] = {
    "abs": torch.abs,
    "tanh": torch.tanh,
    "silu": torch.nn.functional.silu,
    "None": None,
}

__all__ = [
    "AtomicEnergiesBlock",
    "RadialEmbeddingBlock",
    "LinearNodeEmbeddingBlock",
    "LinearReadoutBlock",
    "EquivariantProductBasisBlock",
    "ScaleShiftBlock",
    "LinearDipoleReadoutBlock",
    "NonLinearDipoleReadoutBlock",
    "InteractionBlock",
    "NonLinearReadoutBlock",
    "PolynomialCutoff",
    "BesselBasis",
    "MACE",
    "ScaleShiftMACE",
    "BOTNet",
    "ScaleShiftBOTNet",
    "AtomicDipolesMACE",
    "EnergyDipolesMACE",
    "EnergyForcesLoss",
    "WeightedEnergyForcesLoss",
    "WeightedForcesLoss",
    "WeightedEnergyForcesVirialsLoss",
    "WeightedEnergyForcesStressLoss",
    "DipoleSingleLoss",
    "WeightedEnergyForcesDipoleLoss",
    "SymmetricContraction",
    "interaction_classes",
    "compute_mean_std_atomic_inter_energy",
    "compute_avg_num_neighbors",
    "compute_fixed_charge_dipole",
]
