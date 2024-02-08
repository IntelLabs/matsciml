###########################################################################################
# Implementation of MACE models and other models based E(3)-Equivariant MPNNs
# Authors: Ilyes Batatia, Gregor Simm
# Integrated into matsciml by Vaibhav Bihani, Sajid Mannan
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import torch
import torch_geometric as pyg
from e3nn import o3
from e3nn.util.jit import compile_mode

from matsciml.common.types import BatchDict, DataDict, Embeddings
from matsciml.models.base import AbstractPyGModel
from matsciml.models.pyg.mace import tools
from matsciml.models.pyg.mace.data import AtomicData, get_neighborhood
from matsciml.models.pyg.mace.modules.blocks import (
    AtomicEnergiesBlock,
    EquivariantProductBasisBlock,
    InteractionBlock,
    LinearDipoleReadoutBlock,
    LinearNodeEmbeddingBlock,
    LinearReadoutBlock,
    NonLinearDipoleReadoutBlock,
    NonLinearReadoutBlock,
    RadialEmbeddingBlock,
    ScaleShiftBlock,
)
from matsciml.models.pyg.mace.modules.utils import (
    compute_fixed_charge_dipole,
    compute_forces,
    get_edge_vectors_and_lengths,
    get_outputs,
    get_symmetric_displacement,
)
from matsciml.models.pyg.mace.tools import atomic_numbers_to_indices, to_one_hot
from matsciml.models.pyg.mace.tools.scatter import scatter_sum


@compile_mode("script")
class MACE(AbstractPyGModel):
    def __init__(
        self,
        r_max: float,
        num_bessel: int,
        num_polynomial_cutoff: int,
        max_ell: int,
        interaction_cls: type[InteractionBlock],
        interaction_cls_first: type[InteractionBlock],
        num_interactions: int,
        num_elements: int,
        hidden_irreps: o3.Irreps,
        MLP_irreps: o3.Irreps,
        atomic_energies: np.ndarray,
        avg_num_neighbors: float,
        atomic_numbers: list[int],
        correlation: int,
        gate: Callable | None,
    ):
        super().__init__(atom_embedding_dim=16)
        self.register_buffer(
            "atomic_numbers",
            torch.tensor(atomic_numbers, dtype=torch.int64),
        )
        self.register_buffer("r_max", torch.tensor(r_max, dtype=torch.float64))
        self.register_buffer(
            "num_interactions",
            torch.tensor(num_interactions, dtype=torch.int64),
        )
        # Embedding
        node_attr_irreps = o3.Irreps([(num_elements, (0, 1))])
        node_feats_irreps = o3.Irreps([(hidden_irreps.count(o3.Irrep(0, 1)), (0, 1))])
        self.node_embedding = LinearNodeEmbeddingBlock(
            irreps_in=node_attr_irreps,
            irreps_out=node_feats_irreps,
        )
        self.radial_embedding = RadialEmbeddingBlock(
            r_max=r_max,
            num_bessel=num_bessel,
            num_polynomial_cutoff=num_polynomial_cutoff,
        )
        edge_feats_irreps = o3.Irreps(f"{self.radial_embedding.out_dim}x0e")

        sh_irreps = o3.Irreps.spherical_harmonics(max_ell)
        num_features = hidden_irreps.count(o3.Irrep(0, 1))
        interaction_irreps = (sh_irreps * num_features).sort()[0].simplify()
        self.spherical_harmonics = o3.SphericalHarmonics(
            sh_irreps,
            normalize=True,
            normalization="component",
        )

        # Interactions and readout
        self.atomic_energies_fn = AtomicEnergiesBlock(atomic_energies)

        inter = interaction_cls_first(
            node_attrs_irreps=node_attr_irreps,
            node_feats_irreps=node_feats_irreps,
            edge_attrs_irreps=sh_irreps,
            edge_feats_irreps=edge_feats_irreps,
            target_irreps=interaction_irreps,
            hidden_irreps=hidden_irreps,
            avg_num_neighbors=avg_num_neighbors,
        )
        self.interactions = torch.nn.ModuleList([inter])

        # Use the appropriate self connection at the first layer for proper E0
        use_sc_first = False
        if "Residual" in str(interaction_cls_first):
            use_sc_first = True

        node_feats_irreps_out = inter.target_irreps
        prod = EquivariantProductBasisBlock(
            node_feats_irreps=node_feats_irreps_out,
            target_irreps=hidden_irreps,
            correlation=correlation,
            num_elements=num_elements,
            use_sc=use_sc_first,
        )
        self.products = torch.nn.ModuleList([prod])

        self.readouts = torch.nn.ModuleList()
        self.readouts.append(LinearReadoutBlock(hidden_irreps))

        for i in range(num_interactions - 1):
            if i == num_interactions - 2:
                hidden_irreps_out = str(
                    hidden_irreps[0],
                )  # Select only scalars for last layer
            else:
                hidden_irreps_out = hidden_irreps
            inter = interaction_cls(
                node_attrs_irreps=node_attr_irreps,
                node_feats_irreps=hidden_irreps,
                edge_attrs_irreps=sh_irreps,
                edge_feats_irreps=edge_feats_irreps,
                target_irreps=interaction_irreps,
                hidden_irreps=hidden_irreps_out,
                avg_num_neighbors=avg_num_neighbors,
            )
            self.interactions.append(inter)
            prod = EquivariantProductBasisBlock(
                node_feats_irreps=interaction_irreps,
                target_irreps=hidden_irreps_out,
                correlation=correlation,
                num_elements=num_elements,
                use_sc=True,
            )
            self.products.append(prod)
            if i == num_interactions - 2:
                self.readouts.append(
                    NonLinearReadoutBlock(hidden_irreps_out, MLP_irreps, gate),
                )
            else:
                self.readouts.append(LinearReadoutBlock(hidden_irreps))

    def _forward(
        self,
        data: dict[str, torch.Tensor],
        training: bool = False,
        compute_force: bool = True,
        compute_virials: bool = False,
        compute_stress: bool = False,
        compute_displacement: bool = False,
    ) -> dict[str, torch.Tensor | None]:
        # Setup
        data["positions"].requires_grad_(True)
        num_graphs = data["ptr"].numel() - 1
        displacement = torch.zeros(
            (num_graphs, 3, 3),
            dtype=data["positions"].dtype,
            device=data["positions"].device,
        )
        if compute_virials or compute_stress or compute_displacement:
            (
                data["positions"],
                data["shifts"],
                displacement,
            ) = get_symmetric_displacement(
                positions=data["positions"],
                unit_shifts=data["unit_shifts"],
                cell=data["cell"],
                edge_index=data["edge_index"],
                num_graphs=num_graphs,
                batch=data["batch"],
            )

        # Atomic energies
        node_e0 = self.atomic_energies_fn(data["node_attrs"])
        e0 = scatter_sum(
            src=node_e0,
            index=data["batch"],
            dim=-1,
            dim_size=num_graphs,
        )  # [n_graphs,]

        # Embeddings
        node_feats = self.node_embedding(data["node_attrs"])
        vectors, lengths = get_edge_vectors_and_lengths(
            positions=data["positions"],
            edge_index=data["edge_index"],
            shifts=data["shifts"],
        )
        edge_attrs = self.spherical_harmonics(vectors)
        edge_feats = self.radial_embedding(lengths)

        # Interactions
        energies = [e0]
        node_energies_list = [node_e0]
        for interaction, product, readout in zip(
            self.interactions,
            self.products,
            self.readouts,
        ):
            node_feats, sc = interaction(
                node_attrs=data["node_attrs"],
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=data["edge_index"],
            )
            node_feats = product(
                node_feats=node_feats,
                sc=sc,
                node_attrs=data["node_attrs"],
            )
            node_energies = readout(node_feats).squeeze(-1)  # [n_nodes, ]
            energy = scatter_sum(
                src=node_energies,
                index=data["batch"],
                dim=-1,
                dim_size=num_graphs,
            )  # [n_graphs,]
            energies.append(energy)
            node_energies_list.append(node_energies)

        # Sum over energy contributions
        contributions = torch.stack(energies, dim=-1)
        total_energy = torch.sum(contributions, dim=-1)  # [n_graphs, ]
        node_energy_contributions = torch.stack(node_energies_list, dim=-1)
        node_energy = torch.sum(node_energy_contributions, dim=-1)  # [n_nodes, ]
        # Outputs
        forces, virials, stress = get_outputs(
            energy=total_energy,
            positions=data["positions"],
            displacement=displacement,
            cell=data["cell"],
            training=training,
            compute_force=compute_force,
            compute_virials=compute_virials,
            compute_stress=compute_stress,
        )

        return {
            "energy": total_energy,
            "node_energy": node_energy,
            "contributions": contributions,
            "forces": forces,
            "virials": virials,
            "stress": stress,
            "displacement": displacement,
        }


@compile_mode("script")
class ScaleShiftMACE(MACE):
    def __init__(
        self,
        atomic_inter_scale: float,
        atomic_inter_shift: float,
        training: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.scale_shift = ScaleShiftBlock(
            scale=atomic_inter_scale,
            shift=atomic_inter_shift,
        )

    def _forward(
        self,
        data: dict[str, torch.Tensor],
        compute_force: bool = True,
        compute_virials: bool = False,
        compute_stress: bool = False,
        compute_displacement: bool = False,
    ) -> dict[str, torch.Tensor | None]:
        # Setup
        data["positions"].requires_grad_(True)

        num_graphs = data["ptr"].numel() - 1
        displacement = torch.zeros(
            (num_graphs, 3, 3),
            dtype=data["positions"].dtype,
            device=data["positions"].device,
        )

        if compute_virials or compute_stress or compute_displacement:
            (
                data["positions"],
                data["shifts"],
                displacement,
            ) = get_symmetric_displacement(
                positions=data["positions"],
                unit_shifts=data["unit_shifts"],
                cell=data["cell"],
                edge_index=data["edge_index"],
                num_graphs=num_graphs,
                batch=data["batch"],
            )

        # Atomic energies
        node_e0 = self.atomic_energies_fn(data["node_attrs"])
        e0 = scatter_sum(
            src=node_e0,
            index=data["batch"],
            dim=-1,
            dim_size=num_graphs,
        )  # [n_graphs,]

        # Embeddings
        node_feats = self.node_embedding(data["node_attrs"])
        vectors, lengths = get_edge_vectors_and_lengths(
            positions=data["positions"],
            edge_index=data["edge_index"],
            shifts=data["shifts"],
        )
        edge_attrs = self.spherical_harmonics(vectors)
        edge_feats = self.radial_embedding(lengths)

        # Interactions
        node_es_list = []
        for interaction, product, readout in zip(
            self.interactions,
            self.products,
            self.readouts,
        ):
            node_feats, sc = interaction(
                node_attrs=data["node_attrs"],
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=data["edge_index"],
            )
            node_feats = product(
                node_feats=node_feats,
                sc=sc,
                node_attrs=data["node_attrs"],
            )
            node_es_list.append(readout(node_feats).squeeze(-1))  # {[n_nodes, ], }

        # Sum over interactions
        node_inter_es = torch.sum(
            torch.stack(node_es_list, dim=0),
            dim=0,
        )  # [n_nodes, ]
        node_inter_es = self.scale_shift(node_inter_es)

        # Sum over nodes in graph
        inter_e = scatter_sum(
            src=node_inter_es,
            index=data["batch"],
            dim=-1,
            dim_size=num_graphs,
        )  # [n_graphs,]

        # Add E_0 and (scaled) interaction energy
        total_energy = e0 + inter_e
        node_energy = node_e0 + node_inter_es
        forces, virials, stress = get_outputs(
            energy=total_energy,
            positions=data["positions"],
            displacement=displacement,
            cell=data["cell"],
            training=self.training,
            compute_force=compute_force,
            compute_virials=compute_virials,
            compute_stress=compute_stress,
        )
        output = {
            "energy": total_energy,
            "node_energy": node_energy,
            "interaction_energy": inter_e,
            "forces": forces,
            # "virials": virials,
            # "stress": stress,
            "displacement": displacement,
            "node_feats": node_feats,
            "edge_feats": edge_feats,
        }

        return Embeddings(
            {"energy": output["energy"].reshape(-1, 1), "force": output["forces"]},
        )

    def read_batch(self, batch: BatchDict) -> DataDict:
        """
        Extract PyG structure and features to pass into the model.

        More complicated models can override this method to extract out edge and
        graph features as well.

        Parameters
        ----------
        batch : BatchDict
            Batch of data to process.


        Returns
        -------
        DataDict
            Dictionary of input features to pass into the model
        """
        assert (
            "graph" in batch
        ), f"Model {self.__class__.__name__} expects graph structures, but 'graph' key was not found in batch."
        graph = batch.get("graph")
        pbc = batch.get("pbc")
        data = {"cell": batch.get("cell"), "energy": batch.get("energy")}

        assert isinstance(
            graph,
            (pyg.data.Data, pyg.data.Batch),
        ), f"Model {self.__class__.__name__} expects PyG graphs, but data in 'graph' key is type {type(graph)}"

        for key in ["ptr", "batch", "edge_feats", "graph_feats"]:
            data[key] = getattr(graph, key, None)

        data["positions"] = getattr(graph, "pos")
        data["forces"] = getattr(graph, "force", None)
        # Charges default to 0 instead of None if not found
        data["charges"] = getattr(batch, "charges", torch.zeros_like(data["batch"]))

        data["energy_weight"] = torch.ones((data["positions"].shape[0],))
        data["forces_weight"] = torch.ones((data["positions"].shape[0],))

        data["stress"] = torch.ones((data["positions"].shape[0], 3, 3))
        data["stress_weights"] = torch.ones((data["positions"].shape[0],))

        data["virials"] = torch.ones((data["positions"].shape[0], 3, 3))
        data["virials_weights"] = torch.ones((data["positions"].shape[0],))

        data["weights"] = torch.ones((data["positions"].shape[0],))

        atomic_numbers: torch.Tensor = getattr(graph, "atomic_numbers")
        z_table = tools.get_atomic_number_table_from_zs(atomic_numbers.numpy())

        indices = atomic_numbers_to_indices(atomic_numbers, z_table=z_table)
        data["node_attrs"] = to_one_hot(
            torch.tensor(indices, dtype=torch.long).unsqueeze(-1),
            num_classes=len(z_table),
        )

        shifts = []
        edge_index = []
        unit_shifts = []
        b_sz = data["ptr"].numel() - 1
        pos_k = data["positions"].reshape(b_sz, -1, 3)
        cell_k = data["cell"].reshape(b_sz, 3, 3)
        for k in range(b_sz):
            pbc_tensor = pbc[k] == 1
            pbc_tuple = (
                pbc_tensor[0].item(),
                pbc_tensor[1].item(),
                pbc_tensor[2].item(),
            )
            edge_index_k, shifts_k, unit_shifts_k = get_neighborhood(
                pos_k[k].numpy(),
                cutoff=self.r_max.item(),
                pbc=pbc_tuple,
                cell=cell_k[k],
            )
            shifts += [torch.Tensor(shifts_k)]
            edge_index += [(torch.Tensor(edge_index_k) + 83 * k).T.to(torch.int64)]
            unit_shifts += [torch.Tensor(unit_shifts_k)]

        data["shifts"] = torch.concatenate(shifts)
        data["unit_shifts"] = torch.concatenate(unit_shifts)
        data["edge_index"] = torch.concatenate(edge_index).T
        return {"data": data}

    def read_batch_size(self, batch: BatchDict) -> int:
        graph = batch["graph"]
        return graph.num_graphs


@compile_mode("script")
class ScaleShiftMACE1(MACE):
    def __init__(
        self,
        atomic_inter_scale: float,
        atomic_inter_shift: float,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.scale_shift = ScaleShiftBlock(
            scale=atomic_inter_scale,
            shift=atomic_inter_shift,
        )

    def _forward(
        self,
        data: dict[str, torch.Tensor],
        training: bool = False,
    ) -> dict[str, torch.Tensor | None]:
        # Setup
        data["positions"].requires_grad_(True)
        num_graphs = data["ptr"].numel() - 1

        # Atomic energies
        node_e0 = self.atomic_energies_fn(data["node_attrs"])
        e0 = scatter_sum(
            src=node_e0,
            index=data["batch"],
            dim=-1,
            dim_size=num_graphs,
        )  # [n_graphs,]

        # Embeddings
        node_feats = self.node_embedding(data["node_attrs"])
        vectors, lengths = get_edge_vectors_and_lengths(
            positions=data["positions"],
            edge_index=data["edge_index"],
            shifts=data["shifts"],
        )
        edge_attrs = self.spherical_harmonics(vectors)
        edge_feats = self.radial_embedding(lengths)

        # Interactions
        node_es_list = []
        node_feats_list = []
        for interaction, product, readout in zip(
            self.interactions,
            self.products,
            self.readouts,
        ):
            node_feats, sc = interaction(
                node_attrs=data["node_attrs"],
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=data["edge_index"],
            )
            node_feats = product(
                node_feats=node_feats,
                sc=sc,
                node_attrs=data["node_attrs"],
            )
            node_feats_list.append(node_feats)
            node_es_list.append(readout(node_feats).squeeze(-1))  # {[n_nodes, ], }

        # Sum over interactions
        node_inter_es = torch.sum(
            torch.stack(node_es_list, dim=0),
            dim=0,
        )  # [n_nodes, ]
        node_inter_es = self.scale_shift(node_inter_es)

        # Sum over nodes in graph
        inter_e = scatter_sum(
            src=node_inter_es,
            index=data["batch"],
            dim=-1,
            dim_size=num_graphs,
        )  # [n_graphs,]

        # Add E_0 and (scaled) interaction energy
        total_energy = e0 + inter_e
        node_energy = node_e0 + node_inter_es

        output = {
            "energy": total_energy,
            "node_energy": node_energy,
            "interaction_energy": inter_e,
            # "forces": forces,
            # "virials": virials,
            # "stress": stress,
            "node_feats_list": torch.stack(node_feats_list[:-1], dim=0),
            "edge_feats": edge_feats,
        }

        return Embeddings(None, output["node_feats_list"])

    def read_batch(self, batch: BatchDict) -> DataDict:
        """
        Extract PyG structure and features to pass into the model.

        More complicated models can override this method to extract out edge and
        graph features as well.

        Parameters
        ----------
        batch : BatchDict
            Batch of data to process.


        Returns
        -------
        DataDict
            Dictionary of input features to pass into the model
        """
        assert (
            "graph" in batch
        ), f"Model {self.__class__.__name__} expects graph structures, but 'graph' key was not found in batch."
        graph = batch.get("graph")
        pbc = batch.get("pbc")
        data = {"cell": batch.get("cell"), "energy": batch.get("energy")}

        assert isinstance(
            graph,
            (pyg.data.Data, pyg.data.Batch),
        ), f"Model {self.__class__.__name__} expects PyG graphs, but data in 'graph' key is type {type(graph)}"

        for key in ["ptr", "batch", "edge_feats", "graph_feats"]:
            data[key] = getattr(graph, key, None)

        data["positions"] = getattr(graph, "pos")
        data["forces"] = getattr(graph, "force", None)
        # Charges default to 0 instead of None if not found
        data["charges"] = getattr(batch, "charges", torch.zeros_like(data["batch"]))

        data["energy_weight"] = torch.ones((data["positions"].shape[0],))
        data["forces_weight"] = torch.ones((data["positions"].shape[0],))

        data["stress"] = torch.ones((data["positions"].shape[0], 3, 3))
        data["stress_weights"] = torch.ones((data["positions"].shape[0],))

        data["virials"] = torch.ones((data["positions"].shape[0], 3, 3))
        data["virials_weights"] = torch.ones((data["positions"].shape[0],))

        data["weights"] = torch.ones((data["positions"].shape[0],))

        atomic_numbers: torch.Tensor = getattr(graph, "atomic_numbers")
        z_table = tools.get_atomic_number_table_from_zs(atomic_numbers.numpy())

        indices = atomic_numbers_to_indices(atomic_numbers, z_table=z_table)
        data["node_attrs"] = to_one_hot(
            torch.tensor(indices, dtype=torch.long).unsqueeze(-1),
            num_classes=len(z_table),
        )

        shifts = []
        edge_index = []
        unit_shifts = []
        b_sz = data["ptr"].numel() - 1
        pos_k = data["positions"].reshape(b_sz, -1, 3)
        cell_k = data["cell"].reshape(b_sz, 3, 3)
        for k in range(b_sz):
            pbc_tensor = pbc[k] == 1
            pbc_tuple = (
                pbc_tensor[0].item(),
                pbc_tensor[1].item(),
                pbc_tensor[2].item(),
            )
            edge_index_k, shifts_k, unit_shifts_k = get_neighborhood(
                pos_k[k].numpy(),
                cutoff=self.r_max.item(),
                pbc=pbc_tuple,
                cell=cell_k[k],
            )
            shifts += [torch.Tensor(shifts_k)]
            edge_index += [torch.Tensor(edge_index_k).T.to(torch.int64)]
            unit_shifts += [torch.Tensor(unit_shifts_k)]

        data["shifts"] = torch.concatenate(shifts)
        data["unit_shifts"] = torch.concatenate(unit_shifts)
        data["edge_index"] = torch.concatenate(edge_index).T

        return {"data": data}

    def read_batch_size(self, batch: BatchDict) -> int:
        graph = batch["graph"]
        return graph.num_graphs


class BOTNet(torch.nn.Module):
    def __init__(
        self,
        r_max: float,
        num_bessel: int,
        num_polynomial_cutoff: int,
        max_ell: int,
        interaction_cls: type[InteractionBlock],
        interaction_cls_first: type[InteractionBlock],
        num_interactions: int,
        num_elements: int,
        hidden_irreps: o3.Irreps,
        MLP_irreps: o3.Irreps,
        atomic_energies: np.ndarray,
        gate: Callable | None,
        avg_num_neighbors: float,
        atomic_numbers: list[int],
    ):
        super().__init__()
        self.r_max = r_max
        self.atomic_numbers = atomic_numbers
        # Embedding
        node_attr_irreps = o3.Irreps([(num_elements, (0, 1))])
        node_feats_irreps = o3.Irreps([(hidden_irreps.count(o3.Irrep(0, 1)), (0, 1))])
        self.node_embedding = LinearNodeEmbeddingBlock(
            irreps_in=node_attr_irreps,
            irreps_out=node_feats_irreps,
        )
        self.radial_embedding = RadialEmbeddingBlock(
            r_max=r_max,
            num_bessel=num_bessel,
            num_polynomial_cutoff=num_polynomial_cutoff,
        )
        edge_feats_irreps = o3.Irreps(f"{self.radial_embedding.out_dim}x0e")

        sh_irreps = o3.Irreps.spherical_harmonics(max_ell)
        self.spherical_harmonics = o3.SphericalHarmonics(
            sh_irreps,
            normalize=True,
            normalization="component",
        )

        # Interactions and readouts
        self.atomic_energies_fn = AtomicEnergiesBlock(atomic_energies)

        self.interactions = torch.nn.ModuleList()
        self.readouts = torch.nn.ModuleList()

        inter = interaction_cls_first(
            node_attrs_irreps=node_attr_irreps,
            node_feats_irreps=node_feats_irreps,
            edge_attrs_irreps=sh_irreps,
            edge_feats_irreps=edge_feats_irreps,
            target_irreps=hidden_irreps,
            avg_num_neighbors=avg_num_neighbors,
        )
        self.interactions.append(inter)
        self.readouts.append(LinearReadoutBlock(inter.irreps_out))

        for i in range(num_interactions - 1):
            inter = interaction_cls(
                node_attrs_irreps=node_attr_irreps,
                node_feats_irreps=inter.irreps_out,
                edge_attrs_irreps=sh_irreps,
                edge_feats_irreps=edge_feats_irreps,
                target_irreps=hidden_irreps,
                avg_num_neighbors=avg_num_neighbors,
            )
            self.interactions.append(inter)
            if i == num_interactions - 2:
                self.readouts.append(
                    NonLinearReadoutBlock(inter.irreps_out, MLP_irreps, gate),
                )
            else:
                self.readouts.append(LinearReadoutBlock(inter.irreps_out))

    def forward(self, data: AtomicData, training=False) -> dict[str, Any]:
        # Setup
        data.positions.requires_grad = True

        # Atomic energies
        node_e0 = self.atomic_energies_fn(data.node_attrs)
        e0 = scatter_sum(
            src=node_e0,
            index=data.batch,
            dim=-1,
            dim_size=data.num_graphs,
        )  # [n_graphs,]

        # Embeddings
        node_feats = self.node_embedding(data.node_attrs)
        vectors, lengths = get_edge_vectors_and_lengths(
            positions=data.positions,
            edge_index=data.edge_index,
            shifts=data.shifts,
        )
        edge_attrs = self.spherical_harmonics(vectors)
        edge_feats = self.radial_embedding(lengths)

        # Interactions
        energies = [e0]
        for interaction, readout in zip(self.interactions, self.readouts):
            node_feats = interaction(
                node_attrs=data.node_attrs,
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=data.edge_index,
            )
            node_energies = readout(node_feats).squeeze(-1)  # [n_nodes, ]
            energy = scatter_sum(
                src=node_energies,
                index=data.batch,
                dim=-1,
                dim_size=data.num_graphs,
            )  # [n_graphs,]
            energies.append(energy)

        # Sum over energy contributions
        contributions = torch.stack(energies, dim=-1)
        total_energy = torch.sum(contributions, dim=-1)  # [n_graphs, ]

        output = {
            "energy": total_energy,
            "contributions": contributions,
            "forces": compute_forces(
                energy=total_energy,
                positions=data.positions,
                training=training,
            ),
        }

        return output


class ScaleShiftBOTNet(BOTNet):
    def __init__(
        self,
        atomic_inter_scale: float,
        atomic_inter_shift: float,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.scale_shift = ScaleShiftBlock(
            scale=atomic_inter_scale,
            shift=atomic_inter_shift,
        )

    def forward(self, data: AtomicData, training=False) -> dict[str, Any]:
        # Setup
        data.positions.requires_grad = True

        # Atomic energies
        node_e0 = self.atomic_energies_fn(data.node_attrs)
        e0 = scatter_sum(
            src=node_e0,
            index=data.batch,
            dim=-1,
            dim_size=data.num_graphs,
        )  # [n_graphs,]

        # Embeddings
        node_feats = self.node_embedding(data.node_attrs)
        vectors, lengths = get_edge_vectors_and_lengths(
            positions=data.positions,
            edge_index=data.edge_index,
            shifts=data.shifts,
        )
        edge_attrs = self.spherical_harmonics(vectors)
        edge_feats = self.radial_embedding(lengths)

        # Interactions
        node_es_list = []
        for interaction, readout in zip(self.interactions, self.readouts):
            node_feats = interaction(
                node_attrs=data.node_attrs,
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=data.edge_index,
            )

            node_es_list.append(readout(node_feats).squeeze(-1))  # {[n_nodes, ], }

        # Sum over interactions
        node_inter_es = torch.sum(
            torch.stack(node_es_list, dim=0),
            dim=0,
        )  # [n_nodes, ]
        node_inter_es = self.scale_shift(node_inter_es)

        # Sum over nodes in graph
        inter_e = scatter_sum(
            src=node_inter_es,
            index=data.batch,
            dim=-1,
            dim_size=data.num_graphs,
        )  # [n_graphs,]

        # Add E_0 and (scaled) interaction energy
        total_e = e0 + inter_e

        output = {
            "energy": total_e,
            "forces": compute_forces(
                energy=inter_e,
                positions=data.positions,
                training=training,
            ),
        }

        return output


class AtomicDipolesMACE(torch.nn.Module):
    def __init__(
        self,
        r_max: float,
        num_bessel: int,
        num_polynomial_cutoff: int,
        max_ell: int,
        interaction_cls: type[InteractionBlock],
        interaction_cls_first: type[InteractionBlock],
        num_interactions: int,
        num_elements: int,
        hidden_irreps: o3.Irreps,
        MLP_irreps: o3.Irreps,
        avg_num_neighbors: float,
        atomic_numbers: list[int],
        correlation: int,
        gate: Callable | None,
        atomic_energies: None
        | (None),  # Just here to make it compatible with energy models, MUST be None
    ):
        super().__init__()
        assert atomic_energies is None
        self.r_max = r_max
        self.atomic_numbers = atomic_numbers

        # Embedding
        node_attr_irreps = o3.Irreps([(num_elements, (0, 1))])
        node_feats_irreps = o3.Irreps([(hidden_irreps.count(o3.Irrep(0, 1)), (0, 1))])
        self.node_embedding = LinearNodeEmbeddingBlock(
            irreps_in=node_attr_irreps,
            irreps_out=node_feats_irreps,
        )
        self.radial_embedding = RadialEmbeddingBlock(
            r_max=r_max,
            num_bessel=num_bessel,
            num_polynomial_cutoff=num_polynomial_cutoff,
        )
        edge_feats_irreps = o3.Irreps(f"{self.radial_embedding.out_dim}x0e")

        sh_irreps = o3.Irreps.spherical_harmonics(max_ell)
        num_features = hidden_irreps.count(o3.Irrep(0, 1))
        interaction_irreps = (sh_irreps * num_features).sort()[0].simplify()
        self.spherical_harmonics = o3.SphericalHarmonics(
            sh_irreps,
            normalize=True,
            normalization="component",
        )

        # Interactions and readouts
        inter = interaction_cls_first(
            node_attrs_irreps=node_attr_irreps,
            node_feats_irreps=node_feats_irreps,
            edge_attrs_irreps=sh_irreps,
            edge_feats_irreps=edge_feats_irreps,
            target_irreps=hidden_irreps,
            hidden_irreps=hidden_irreps,
            avg_num_neighbors=avg_num_neighbors,
        )
        self.interactions = torch.nn.ModuleList([inter])

        # Use the appropriate self connection at the first layer
        use_sc_first = False
        if "Residual" in str(interaction_cls_first):
            use_sc_first = True

        node_feats_irreps_out = inter.target_irreps
        prod = EquivariantProductBasisBlock(
            node_feats_irreps=node_feats_irreps_out,
            target_irreps=hidden_irreps,
            correlation=correlation,
            num_elements=num_elements,
            use_sc=use_sc_first,
        )
        self.products = torch.nn.ModuleList([prod])

        self.readouts = torch.nn.ModuleList()
        self.readouts.append(LinearDipoleReadoutBlock(hidden_irreps, dipole_only=True))

        for i in range(num_interactions - 1):
            if i == num_interactions - 2:
                assert (
                    len(hidden_irreps) > 1
                ), "To predict dipoles use at least l=1 hidden_irreps"
                hidden_irreps_out = str(
                    hidden_irreps[1],
                )  # Select only l=1 vectors for last layer
            else:
                hidden_irreps_out = hidden_irreps
            inter = interaction_cls(
                node_attrs_irreps=node_attr_irreps,
                node_feats_irreps=inter.irreps_out,
                edge_attrs_irreps=sh_irreps,
                edge_feats_irreps=edge_feats_irreps,
                target_irreps=interaction_irreps,
                hidden_irreps=hidden_irreps_out,
                avg_num_neighbors=avg_num_neighbors,
            )
            self.interactions.append(inter)
            prod = EquivariantProductBasisBlock(
                node_feats_irreps=interaction_irreps,
                target_irreps=hidden_irreps_out,
                correlation=correlation,
                num_elements=num_elements,
                use_sc=True,
            )
            self.products.append(prod)
            if i == num_interactions - 2:
                self.readouts.append(
                    NonLinearDipoleReadoutBlock(
                        hidden_irreps_out,
                        MLP_irreps,
                        gate,
                        dipole_only=True,
                    ),
                )
            else:
                self.readouts.append(
                    LinearDipoleReadoutBlock(hidden_irreps, dipole_only=True),
                )

    def forward(
        self,
        data: AtomicData,
        training=False,
        compute_force: bool = False,
        compute_virials: bool = False,
        compute_stress: bool = False,
    ) -> dict[str, Any]:
        assert compute_force is False
        assert compute_virials is False
        assert compute_stress is False
        # Setup
        data.positions.requires_grad = True
        if not training:
            for p in self.parameters():
                p.requires_grad = False
        else:
            for p in self.parameters():
                p.requires_grad = True

        # Embeddings
        node_feats = self.node_embedding(data.node_attrs)
        vectors, lengths = get_edge_vectors_and_lengths(
            positions=data.positions,
            edge_index=data.edge_index,
            shifts=data.shifts,
        )
        edge_attrs = self.spherical_harmonics(vectors)
        edge_feats = self.radial_embedding(lengths)

        # Interactions
        dipoles = []
        for interaction, product, readout in zip(
            self.interactions,
            self.products,
            self.readouts,
        ):
            node_feats, sc = interaction(
                node_attrs=data.node_attrs,
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=data.edge_index,
            )
            node_feats = product(
                node_feats=node_feats,
                sc=sc,
                node_attrs=data.node_attrs,
            )
            node_dipoles = readout(node_feats).squeeze(-1)  # [n_nodes,3]
            dipoles.append(node_dipoles)

        # Compute the dipoles
        contributions_dipoles = torch.stack(
            dipoles,
            dim=-1,
        )  # [n_nodes,3,n_contributions]
        atomic_dipoles = torch.sum(contributions_dipoles, dim=-1)  # [n_nodes,3]
        total_dipole = scatter_sum(
            src=atomic_dipoles,
            index=data.batch.unsqueeze(-1),
            dim=0,
            dim_size=data.num_graphs,
        )  # [n_graphs,3]
        baseline = compute_fixed_charge_dipole(
            charges=data.charges,
            positions=data.positions,
            batch=data.batch,
            num_graphs=data.num_graphs,
        )  # [n_graphs,3]
        total_dipole = total_dipole + baseline

        output = {
            "dipole": total_dipole,
            "atomic_dipoles": atomic_dipoles,
        }

        return output


class EnergyDipolesMACE(torch.nn.Module):
    def __init__(
        self,
        r_max: float,
        num_bessel: int,
        num_polynomial_cutoff: int,
        max_ell: int,
        interaction_cls: type[InteractionBlock],
        interaction_cls_first: type[InteractionBlock],
        num_interactions: int,
        num_elements: int,
        hidden_irreps: o3.Irreps,
        MLP_irreps: o3.Irreps,
        avg_num_neighbors: float,
        atomic_numbers: list[int],
        correlation: int,
        gate: Callable | None,
        atomic_energies: np.ndarray | None,
    ):
        super().__init__()
        self.r_max = r_max
        self.atomic_numbers = atomic_numbers
        # Embedding
        node_attr_irreps = o3.Irreps([(num_elements, (0, 1))])
        node_feats_irreps = o3.Irreps([(hidden_irreps.count(o3.Irrep(0, 1)), (0, 1))])
        self.node_embedding = LinearNodeEmbeddingBlock(
            irreps_in=node_attr_irreps,
            irreps_out=node_feats_irreps,
        )
        self.radial_embedding = RadialEmbeddingBlock(
            r_max=r_max,
            num_bessel=num_bessel,
            num_polynomial_cutoff=num_polynomial_cutoff,
        )
        edge_feats_irreps = o3.Irreps(f"{self.radial_embedding.out_dim}x0e")

        sh_irreps = o3.Irreps.spherical_harmonics(max_ell)
        num_features = hidden_irreps.count(o3.Irrep(0, 1))
        interaction_irreps = (sh_irreps * num_features).sort()[0].simplify()
        self.spherical_harmonics = o3.SphericalHarmonics(
            sh_irreps,
            normalize=True,
            normalization="component",
        )

        # Interactions and readouts
        self.atomic_energies_fn = AtomicEnergiesBlock(atomic_energies)

        inter = interaction_cls_first(
            node_attrs_irreps=node_attr_irreps,
            node_feats_irreps=node_feats_irreps,
            edge_attrs_irreps=sh_irreps,
            edge_feats_irreps=edge_feats_irreps,
            target_irreps=hidden_irreps,
            hidden_irreps=hidden_irreps,
            avg_num_neighbors=avg_num_neighbors,
        )
        self.interactions = torch.nn.ModuleList([inter])

        # Use the appropriate self connection at the first layer
        use_sc_first = False
        if "Residual" in str(interaction_cls_first):
            use_sc_first = True

        node_feats_irreps_out = inter.target_irreps
        prod = EquivariantProductBasisBlock(
            node_feats_irreps=node_feats_irreps_out,
            target_irreps=hidden_irreps,
            correlation=correlation,
            num_elements=num_elements,
            use_sc=use_sc_first,
        )
        self.products = torch.nn.ModuleList([prod])

        self.readouts = torch.nn.ModuleList()
        self.readouts.append(LinearDipoleReadoutBlock(hidden_irreps, dipole_only=False))

        for i in range(num_interactions - 1):
            if i == num_interactions - 2:
                assert (
                    len(hidden_irreps) > 1
                ), "To predict dipoles use at least l=1 hidden_irreps"
                hidden_irreps_out = str(
                    hidden_irreps[:2],
                )  # Select scalars and l=1 vectors for last layer
            else:
                hidden_irreps_out = hidden_irreps
            inter = interaction_cls(
                node_attrs_irreps=node_attr_irreps,
                node_feats_irreps=inter.irreps_out,
                edge_attrs_irreps=sh_irreps,
                edge_feats_irreps=edge_feats_irreps,
                target_irreps=interaction_irreps,
                hidden_irreps=hidden_irreps_out,
                avg_num_neighbors=avg_num_neighbors,
            )
            self.interactions.append(inter)
            prod = EquivariantProductBasisBlock(
                node_feats_irreps=interaction_irreps,
                target_irreps=hidden_irreps_out,
                correlation=correlation,
                num_elements=num_elements,
                use_sc=True,
            )
            self.products.append(prod)
            if i == num_interactions - 2:
                self.readouts.append(
                    NonLinearDipoleReadoutBlock(
                        hidden_irreps_out,
                        MLP_irreps,
                        gate,
                        dipole_only=False,
                    ),
                )
            else:
                self.readouts.append(
                    LinearDipoleReadoutBlock(hidden_irreps, dipole_only=False),
                )

    def forward(
        self,
        data: AtomicData,
        training=False,
        compute_force: bool = True,
        compute_virials: bool = False,
        compute_stress: bool = False,
    ) -> dict[str, Any]:
        # dipoles and virials / stress not supported simultaneously
        assert compute_virials is False
        assert compute_stress is False
        # Setup
        data.positions.requires_grad = True
        if not training:
            for p in self.parameters():
                p.requires_grad = False
        else:
            for p in self.parameters():
                p.requires_grad = True

        # Atomic energies
        node_e0 = self.atomic_energies_fn(data.node_attrs)
        e0 = scatter_sum(
            src=node_e0,
            index=data.batch,
            dim=-1,
            dim_size=data.num_graphs,
        )  # [n_graphs,]

        # Embeddings
        node_feats = self.node_embedding(data.node_attrs)
        vectors, lengths = get_edge_vectors_and_lengths(
            positions=data.positions,
            edge_index=data.edge_index,
            shifts=data.shifts,
        )
        edge_attrs = self.spherical_harmonics(vectors)
        edge_feats = self.radial_embedding(lengths)

        # Interactions
        energies = [e0]
        dipoles = []
        for interaction, product, readout in zip(
            self.interactions,
            self.products,
            self.readouts,
        ):
            node_feats, sc = interaction(
                node_attrs=data.node_attrs,
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=data.edge_index,
            )
            node_feats = product(
                node_feats=node_feats,
                sc=sc,
                node_attrs=data.node_attrs,
            )
            node_out = readout(node_feats).squeeze(-1)  # [n_nodes, ]
            # node_energies = readout(node_feats).squeeze(-1)  # [n_nodes, ]
            node_energies = node_out[:, 0]
            energy = scatter_sum(
                src=node_energies,
                index=data.batch,
                dim=-1,
                dim_size=data.num_graphs,
            )  # [n_graphs,]
            energies.append(energy)
            # node_dipoles = readout(node_feats).squeeze(-1)  # [n_nodes,3]
            node_dipoles = node_out[:, 1:]
            dipoles.append(node_dipoles)

        # Compute the energies and dipoles
        contributions = torch.stack(energies, dim=-1)
        total_energy = torch.sum(contributions, dim=-1)  # [n_graphs, ]
        contributions_dipoles = torch.stack(
            dipoles,
            dim=-1,
        )  # [n_nodes,3,n_contributions]
        atomic_dipoles = torch.sum(contributions_dipoles, dim=-1)  # [n_nodes,3]
        total_dipole = scatter_sum(
            src=atomic_dipoles,
            index=data.batch.unsqueeze(-1),
            dim=0,
            dim_size=data.num_graphs,
        )  # [n_graphs,3]
        baseline = compute_fixed_charge_dipole(
            charges=data.charges,
            positions=data.positions,
            batch=data.batch,
            num_graphs=data.num_graphs,
        )  # [n_graphs,3]
        total_dipole = total_dipole + baseline

        forces, _, _ = get_outputs(
            energy=total_energy,
            positions=data.positions,
            displacement=None,
            cell=data.cell,
            training=training,
            compute_force=compute_force,
            compute_virials=compute_virials,
            compute_stress=compute_stress,
        )

        output = {
            "energy": total_energy,
            "contributions": contributions,
            "forces": forces,
            "dipole": total_dipole,
            "atomic_dipoles": atomic_dipoles,
        }

        return output
