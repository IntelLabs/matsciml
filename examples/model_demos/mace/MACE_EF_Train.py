# import sys
# sys.path.append(".\matsciml")  #Path to matsciml directory(or matsciml installed as package )
from matsciml.models.base import MaceEnergyForceTask
from matsciml.lightning.data_utils import MatSciMLDataModule
from matsciml.models.pyg.mace.modules.blocks import *
from matsciml.models.pyg.mace.modules.models import ScaleShiftMACE
from matsciml.models.pyg.mace import tools
import lightning.pytorch as pl
from matsciml.models.pyg.mace.tools import atomic_numbers_to_indices, to_one_hot
import e3nn
import argparse

from matsciml.datasets.transforms import PointCloudToGraphTransform

# Atomic Energies table
E0s = {
    1: -13.663181292231226,
    3: -216.78673811801755,
    6: -1029.2809654211628,
    7: -1484.1187695035828,
    8: -2042.0330099956639,
    15: -1537.0898574856286,
    16: -1867.8202267974733,
}


def to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.cpu().detach().numpy()


def compute_mean_std_atomic_inter_energy_and_avg_num_neighbors(
    data_loader: torch.utils.data.DataLoader,
    atomic_energies: np.ndarray,
) -> Tuple[float, float]:
    atomic_energies_fn = AtomicEnergiesBlock(atomic_energies=atomic_energies)

    avg_atom_inter_es_list = []
    avg_num_neighbors_list = []
    for batch in data_loader:
        graph = batch.get("graph")
        atomic_numbers: torch.Tensor = getattr(graph, "atomic_numbers")
        z_table = tools.get_atomic_number_table_from_zs(atomic_numbers.numpy())

        indices = atomic_numbers_to_indices(atomic_numbers, z_table=z_table)
        node_attrs = to_one_hot(
            torch.tensor(indices, dtype=torch.long).unsqueeze(-1),
            num_classes=len(z_table),
        )
        node_e0 = atomic_energies_fn(node_attrs)
        graph_e0s = scatter_sum(
            src=node_e0, index=graph.batch, dim=-1, dim_size=graph.num_graphs
        )
        graph_sizes = graph.ptr[1:] - graph.ptr[:-1]
        avg_atom_inter_es_list.append(
            (batch["energy"] - graph_e0s) / graph_sizes
        )  # {[n_graphs], }
        avg_num_neighbors_list.append(graph.edge_index.numel() / len(atomic_numbers))

    avg_atom_inter_es = torch.cat(avg_atom_inter_es_list)  # [total_n_graphs]
    mean = to_numpy(torch.mean(avg_atom_inter_es)).item()
    std = to_numpy(torch.std(avg_atom_inter_es)).item()
    avg_num_neighbors = torch.mean(torch.Tensor(avg_num_neighbors_list))
    return mean, std, avg_num_neighbors


def main(args):
    # Load Data
    dm = MatSciMLDataModule.from_devset(
        "LiPSDataset",
        dset_kwargs={
            "transforms": [PointCloudToGraphTransform("pyg", cutoff_dist=args.cutoff)]
        },
    )

    dm.setup()
    Train_loader = dm.train_dataloader()
    dataset_iter = iter(Train_loader)
    batch = next(dataset_iter)

    atomic_numbers = torch.unique(batch["graph"]["atomic_numbers"]).numpy()
    atomic_energies = np.array([E0s[i] for i in atomic_numbers])

    atomic_inter_shift, atomic_inter_scale, avg_num_neighbors = (
        compute_mean_std_atomic_inter_energy_and_avg_num_neighbors(
            Train_loader, atomic_energies
        )
    )

    # Load Model
    model_config = dict(
        r_max=args.cutoff,
        num_bessel=args.num_bessel,
        num_polynomial_cutoff=args.num_polynomial_cutoff,
        max_ell=args.Lmax,
        interaction_cls=RealAgnosticResidualInteractionBlock,
        num_interactions=args.num_interactions,
        num_elements=len(atomic_numbers),
        hidden_irreps=e3nn.o3.Irreps(args.hidden_irreps),
        atomic_energies=atomic_energies,
        avg_num_neighbors=avg_num_neighbors,
        atomic_numbers=atomic_numbers,
        correlation=args.correlation_order,
        gate=torch.nn.functional.silu,
        interaction_cls_first=RealAgnosticResidualInteractionBlock,
        MLP_irreps=e3nn.o3.Irreps(args.MLP_irreps),
        atomic_inter_scale=atomic_inter_scale,
        atomic_inter_shift=atomic_inter_shift,
        training=True,
    )

    task = MaceEnergyForceTask(
        encoder_class=ScaleShiftMACE,
        encoder_kwargs=model_config,
        task_keys=["energy", "force"],
        output_kwargs={
            "energy": {
                "block_type": "IdentityOutputBlock",
                "output_dim": 1,
                "hidden_dim": None,
            },
            "force": {
                "block_type": "IdentityOutputBlock",
                "output_dim": 3,
                "hidden_dim": None,
            },
        },
        loss_coeff={"energy": 1.0, "force": 1000.0},
    )

    # Print model
    print(task)

    # Start Training

    trainer = pl.Trainer(max_epochs=args.max_epochs, log_every_n_steps=10)

    trainer.fit(task, datamodule=dm)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MACE Training script")
    parser.add_argument("--cutoff", type=float, default=5.0, help="Neighbor cutoff")
    parser.add_argument("--Lmax", type=int, default=3, help="Spherical harmonic Lmax")
    parser.add_argument(
        "--num_bessel", type=int, default=8, help="Bessel embeding size"
    )
    parser.add_argument(
        "--num_polynomial_cutoff",
        type=int,
        default=5,
        help="Radial basis polynomial cutoff",
    )
    parser.add_argument(
        "--num_interactions", type=int, default=2, help="No. of interaction layers"
    )
    parser.add_argument(
        "--hidden_irreps",
        type=str,
        default="16x0e+16x1o+16x2e",
        help="Hidden Irrep Shape",
    )
    parser.add_argument(
        "--correlation_order", type=int, default=3, help="Correlation Order"
    )
    parser.add_argument(
        "--MLP_irreps",
        type=str,
        default="16x0e",
        help="Irreps of Non-linear readout block",
    )
    parser.add_argument("--max_epochs", type=int, default=1000, help="Max epochs")

    args = parser.parse_args()
    main(args)
