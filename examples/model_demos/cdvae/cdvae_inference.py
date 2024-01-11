# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: MIT License
from __future__ import annotations

import argparse
import os
import sys
import time
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import pytorch_lightning as pl
import torch
from torch.optim import Adam
from torch_geometric.data import Batch
from tqdm import tqdm

try:
    from examples.model_demos.cdvae.cdvae import get_scalers
    from examples.model_demos.cdvae.cdvae_configs import (
        cdvae_config,
        dec_config,
        enc_config,
        mp_config,
    )
    from matsciml.datasets.materials_project import CdvaeLMDBDataset
    from matsciml.lightning.data_utils import MatSciMLDataModule
    from matsciml.models.diffusion_pipeline import GenerationTask
    from matsciml.models.pyg.dimenetpp_wrap_cdvae import DimeNetPlusPlusWrap
    from matsciml.models.pyg.gemnet.decoder import GemNetTDecoder

except:
    dir_path = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(f"{dir_path}/../")
    from examples.model_demos.cdvae.cdvae import get_scalers
    from examples.model_demos.cdvae.cdvae_configs import (
        cdvae_config,
        dec_config,
        enc_config,
        mp_config,
    )
    from matsciml.datasets.materials_project import CdvaeLMDBDataset
    from matsciml.lightning.data_utils import MatSciMLDataModule
    from matsciml.models.diffusion_pipeline import GenerationTask
    from matsciml.models.pyg.dimenetpp_wrap_cdvae import DimeNetPlusPlusWrap
    from matsciml.models.pyg.gemnet.decoder import GemNetTDecoder


def load_model(model_path, data_path, load_data, bs=256):
    data_config = mp_config

    # init dataset-specific params in encoder/decoder
    enc_config["num_targets"] = cdvae_config["latent_dim"]  # data_config['num_targets']
    enc_config["otf_graph"] = data_config["otf_graph"]
    enc_config["readout"] = data_config["readout"]

    cdvae_config["max_atoms"] = data_config["max_atoms"]
    cdvae_config["raidus"] = 12.0
    cdvae_config["teacher_forcing_max_epoch"] = data_config["teacher_forcing_max_epoch"]
    cdvae_config["lattice_scale_method"] = data_config["lattice_scale_method"]

    dm = MatSciMLDataModule(
        dataset=CdvaeLMDBDataset,
        train_path=data_path / "train",
        val_split=data_path / "val",
        test_split=data_path / "test",
        batch_size=bs,
        num_workers=0,
    )
    # Load the data at the setup stage
    dm.setup()
    # Compute scalers for regression targets and lattice parameters
    lattice_scaler, prop_scaler = get_scalers(dm.splits["train"])
    dm.dataset.lattice_scaler = lattice_scaler.copy()
    dm.dataset.scaler = prop_scaler.copy()

    encoder = DimeNetPlusPlusWrap(**enc_config)
    decoder = GemNetTDecoder(**dec_config)
    model = GenerationTask(encoder=encoder, decoder=decoder, **cdvae_config)
    if model_path is not None:
        checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
        model.load_state_dict(checkpoint["state_dict"])

    model.lattice_scaler = lattice_scaler.copy()
    model.scaler = prop_scaler.copy()
    test_loader = dm.test_dataloader()

    return model, test_loader, cdvae_config


def reconstructon(
    loader,
    model,
    ld_kwargs,
    num_evals,
    force_num_atoms=False,
    force_atom_types=False,
    down_sample_traj_step=1,
):
    """
    reconstruct the crystals in <loader>.
    """
    all_frac_coords_stack = []
    all_atom_types_stack = []
    frac_coords = []
    num_atoms = []
    atom_types = []
    lengths = []
    angles = []
    input_data_list = []

    # TODO bugfix due to not instantiating stuff from hydra
    # loader.scaler = model.scaler.copy()
    # loader.lattice_scaler = model.lattice_scaler.copy()

    for idx, batch in enumerate(loader):
        if torch.cuda.is_available():
            batch.cuda()
        print(f"batch {idx} in {len(loader)}")
        batch_all_frac_coords = []
        batch_all_atom_types = []
        batch_frac_coords, batch_num_atoms, batch_atom_types = [], [], []
        batch_lengths, batch_angles = [], []

        # only sample one z, multiple evals for stochasticity in langevin dynamics
        embedding = model.encoder(batch)
        _, _, z = model.encode(embedding)

        for eval_idx in range(num_evals):
            gt_num_atoms = batch.num_atoms if force_num_atoms else None
            gt_atom_types = batch.atom_types if force_atom_types else None
            outputs = model.langevin_dynamics(z, ld_kwargs, gt_num_atoms, gt_atom_types)

            # collect sampled crystals in this batch.
            batch_frac_coords.append(outputs["frac_coords"].detach().cpu())
            batch_num_atoms.append(outputs["num_atoms"].detach().cpu())
            batch_atom_types.append(outputs["atom_types"].detach().cpu())
            batch_lengths.append(outputs["lengths"].detach().cpu())
            batch_angles.append(outputs["angles"].detach().cpu())
            if ld_kwargs.save_traj:
                batch_all_frac_coords.append(
                    outputs["all_frac_coords"][::down_sample_traj_step].detach().cpu(),
                )
                batch_all_atom_types.append(
                    outputs["all_atom_types"][::down_sample_traj_step].detach().cpu(),
                )
        # collect sampled crystals for this z.
        frac_coords.append(torch.stack(batch_frac_coords, dim=0))
        num_atoms.append(torch.stack(batch_num_atoms, dim=0))
        atom_types.append(torch.stack(batch_atom_types, dim=0))
        lengths.append(torch.stack(batch_lengths, dim=0))
        angles.append(torch.stack(batch_angles, dim=0))
        if ld_kwargs.save_traj:
            all_frac_coords_stack.append(torch.stack(batch_all_frac_coords, dim=0))
            all_atom_types_stack.append(torch.stack(batch_all_atom_types, dim=0))
        # Save the ground truth structure
        input_data_list = input_data_list + batch.to_data_list()

    frac_coords = torch.cat(frac_coords, dim=1)
    num_atoms = torch.cat(num_atoms, dim=1)
    atom_types = torch.cat(atom_types, dim=1)
    lengths = torch.cat(lengths, dim=1)
    angles = torch.cat(angles, dim=1)
    if ld_kwargs.save_traj:
        all_frac_coords_stack = torch.cat(all_frac_coords_stack, dim=2)
        all_atom_types_stack = torch.cat(all_atom_types_stack, dim=2)
    input_data_batch = Batch.from_data_list(input_data_list)

    return (
        frac_coords,
        num_atoms,
        atom_types,
        lengths,
        angles,
        all_frac_coords_stack,
        all_atom_types_stack,
        input_data_batch,
    )


def generation(
    model,
    ld_kwargs,
    num_batches_to_sample,
    num_samples_per_z,
    batch_size=512,
    down_sample_traj_step=1,
):
    all_frac_coords_stack = []
    all_atom_types_stack = []
    frac_coords = []
    num_atoms = []
    atom_types = []
    lengths = []
    angles = []

    for z_idx in range(num_batches_to_sample):
        batch_all_frac_coords = []
        batch_all_atom_types = []
        batch_frac_coords, batch_num_atoms, batch_atom_types = [], [], []
        batch_lengths, batch_angles = [], []

        z = torch.randn(batch_size, model.hparams.hidden_dim, device=model.device)

        for sample_idx in range(num_samples_per_z):
            samples = model.langevin_dynamics(z, ld_kwargs)

            # collect sampled crystals in this batch.
            batch_frac_coords.append(samples["frac_coords"].detach().cpu())
            batch_num_atoms.append(samples["num_atoms"].detach().cpu())
            batch_atom_types.append(samples["atom_types"].detach().cpu())
            batch_lengths.append(samples["lengths"].detach().cpu())
            batch_angles.append(samples["angles"].detach().cpu())
            if ld_kwargs.save_traj:
                batch_all_frac_coords.append(
                    samples["all_frac_coords"][::down_sample_traj_step].detach().cpu(),
                )
                batch_all_atom_types.append(
                    samples["all_atom_types"][::down_sample_traj_step].detach().cpu(),
                )

        # collect sampled crystals for this z.
        frac_coords.append(torch.stack(batch_frac_coords, dim=0))
        num_atoms.append(torch.stack(batch_num_atoms, dim=0))
        atom_types.append(torch.stack(batch_atom_types, dim=0))
        lengths.append(torch.stack(batch_lengths, dim=0))
        angles.append(torch.stack(batch_angles, dim=0))
        if ld_kwargs.save_traj:
            all_frac_coords_stack.append(torch.stack(batch_all_frac_coords, dim=0))
            all_atom_types_stack.append(torch.stack(batch_all_atom_types, dim=0))

    frac_coords = torch.cat(frac_coords, dim=1)
    num_atoms = torch.cat(num_atoms, dim=1)
    atom_types = torch.cat(atom_types, dim=1)
    lengths = torch.cat(lengths, dim=1)
    angles = torch.cat(angles, dim=1)
    if ld_kwargs.save_traj:
        all_frac_coords_stack = torch.cat(all_frac_coords_stack, dim=2)
        all_atom_types_stack = torch.cat(all_atom_types_stack, dim=2)
    return (
        frac_coords,
        num_atoms,
        atom_types,
        lengths,
        angles,
        all_frac_coords_stack,
        all_atom_types_stack,
    )


# The optimization component has not been tested as it requires a slightly
# different training procedure with training an additional classifier/regressor
def optimization(
    model,
    ld_kwargs,
    data_loader,
    num_starting_points=100,
    num_gradient_steps=5000,
    lr=1e-3,
    num_saved_crys=10,
):
    if data_loader is not None:
        batch = next(iter(data_loader)).to(model.device)
        _, _, z = model.encode(batch)
        z = z[:num_starting_points].detach().clone()
        z.requires_grad = True
    else:
        z = torch.randn(
            num_starting_points,
            model.hparams.hidden_dim,
            device=model.device,
        )
        z.requires_grad = True

    opt = Adam([z], lr=lr)
    model.freeze()

    all_crystals = []
    interval = num_gradient_steps // (num_saved_crys - 1)
    for i in tqdm(range(num_gradient_steps)):
        opt.zero_grad()
        loss = model.fc_property(z).mean()
        loss.backward()
        opt.step()

        if i % interval == 0 or i == (num_gradient_steps - 1):
            crystals = model.langevin_dynamics(z, ld_kwargs)
            all_crystals.append(crystals)
    return {
        k: torch.cat([d[k] for d in all_crystals]).unsqueeze(0)
        for k in ["frac_coords", "atom_types", "num_atoms", "lengths", "angles"]
    }


def main(args):
    # load_data if do reconstruction.
    model_path = None if args.model_path is None else args.model_path

    data_path = Path(args.data_path)
    model, test_loader, cfg = load_model(
        model_path,
        load_data=("recon" in args.tasks)
        or ("opt" in args.tasks and args.start_from == "data"),
        data_path=data_path,
        bs=args.batch_size,
    )
    ld_kwargs = SimpleNamespace(
        n_step_each=args.n_step_each,
        step_lr=args.step_lr,
        min_sigma=args.min_sigma,
        save_traj=args.save_traj,
        disable_bar=args.disable_bar,
    )

    if torch.cuda.is_available():
        model.to("cuda")

    model_path = Path(os.path.dirname(os.path.realpath(__file__)))
    model_path = model_path / ".." / "outputs"

    if "recon" in args.tasks:
        print("Evaluate model on the reconstruction task.")
        start_time = time.time()
        (
            frac_coords,
            num_atoms,
            atom_types,
            lengths,
            angles,
            all_frac_coords_stack,
            all_atom_types_stack,
            input_data_batch,
        ) = reconstructon(
            test_loader,
            model,
            ld_kwargs,
            args.num_evals,
            args.force_num_atoms,
            args.force_atom_types,
            args.down_sample_traj_step,
        )

        if args.label == "":
            recon_out_name = "eval_recon.pt"
        else:
            recon_out_name = f"eval_recon_{args.label}.pt"

        torch.save(
            {
                "eval_setting": args,
                "input_data_batch": input_data_batch,
                "frac_coords": frac_coords,
                "num_atoms": num_atoms,
                "atom_types": atom_types,
                "lengths": lengths,
                "angles": angles,
                "all_frac_coords_stack": all_frac_coords_stack,
                "all_atom_types_stack": all_atom_types_stack,
                "time": time.time() - start_time,
            },
            model_path / recon_out_name,
        )

    if "gen" in args.tasks:
        print("Evaluate model on the generation task.")
        start_time = time.time()

        (
            frac_coords,
            num_atoms,
            atom_types,
            lengths,
            angles,
            all_frac_coords_stack,
            all_atom_types_stack,
        ) = generation(
            model,
            ld_kwargs,
            args.num_batches_to_samples,
            args.num_evals,
            args.batch_size,
            args.down_sample_traj_step,
        )

        if args.label == "":
            gen_out_name = "eval_gen.pt"
        else:
            gen_out_name = f"eval_gen_{args.label}.pt"

        torch.save(
            {
                "eval_setting": args,
                "frac_coords": frac_coords,
                "num_atoms": num_atoms,
                "atom_types": atom_types,
                "lengths": lengths,
                "angles": angles,
                "all_frac_coords_stack": all_frac_coords_stack,
                "all_atom_types_stack": all_atom_types_stack,
                "time": time.time() - start_time,
            },
            model_path / gen_out_name,
        )

    if "opt" in args.tasks:
        print("Evaluate model on the property optimization task.")
        start_time = time.time()
        if args.start_from == "data":
            loader = test_loader
        else:
            loader = None
        optimized_crystals = optimization(model, ld_kwargs, loader)
        optimized_crystals.update(
            {"eval_setting": args, "time": time.time() - start_time},
        )

        if args.label == "":
            gen_out_name = "eval_opt.pt"
        else:
            gen_out_name = f"eval_opt_{args.label}.pt"
        torch.save(optimized_crystals, model_path / gen_out_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=False)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--tasks", nargs="+", default=["recon", "gen", "opt"])
    parser.add_argument("--n_step_each", default=100, type=int)
    parser.add_argument("--step_lr", default=1e-4, type=float)
    parser.add_argument("--min_sigma", default=0, type=float)
    parser.add_argument("--save_traj", default=False, type=bool)
    parser.add_argument("--disable_bar", default=False, type=bool)
    parser.add_argument("--num_evals", default=1, type=int)
    parser.add_argument("--num_batches_to_samples", default=20, type=int)
    parser.add_argument("--start_from", default="data", type=str)
    parser.add_argument("--batch_size", default=500, type=int)
    parser.add_argument("--force_num_atoms", action="store_true")
    parser.add_argument("--force_atom_types", action="store_true")
    parser.add_argument("--down_sample_traj_step", default=10, type=int)
    parser.add_argument("--label", default="")

    args = parser.parse_args()

    main(args)
