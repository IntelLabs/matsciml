from __future__ import annotations

from matsciml import datasets  # noqa: F401
from matsciml.datasets.transforms import (
    PeriodicPropertiesTransform,
    PointCloudToGraphTransform,
)
from matsciml.common.registry import registry
from matsciml.datasets.utils import make_pymatgen_periodic_structure

import torch
from pymatgen.core import Lattice
from torch_geometric.utils import degree
from tqdm import tqdm
from joblib import dump
import numpy as np

import utils

"""
This script generates a bunch of summary files for datasets included
in ``dataset_mapping``.

Each dataset will have a summary CSV file, which will aggregate all
of the graph and label data for every sample. The idea is you can
then pipeline this into a ``seaborn`` analysis.

The ``utils.py`` module provides some rudimentary data classes that
just makes packing and unpacking data more streamlined.

Currently there is no usable interface, however if there is an interest
in using these routines for new datasets, PRs are welcome.
"""

# this is a hardcoded mapping; change paths to where datasets reside
dataset_mapping = {
    "OQMDDataset": "/data/datasets/matsciml/oqmd/all",
    "NomadDataset": "/data/datasets/matsciml/nomad/all",
    "LiPSDataset": "/data/datasets/matsciml/lips",
    "CMDataset": "/data/datasets/matsciml/carolina_matdb/base/all",
    "MaterialsProjectDataset": "/data/datasets/opencatalyst/mp_data/base",
}

summaries = []

# loop over each dataset, construct graphs with nominal parameters
# then compute the summary statistics
for key, value in dataset_mapping.items():
    dset_class = registry.get_dataset_class(key)
    dset = dset_class(
        lmdb_root_path=value,
        transforms=[
            PeriodicPropertiesTransform(6.0, adaptive_cutoff=True),
            PointCloudToGraphTransform("pyg"),
        ],
    )

    num_data = len(dset)

    summary = utils.DatasetSummary(key, {}, [])

    for index in tqdm(range(num_data), desc=f"Data sample from {key}"):
        sample = dset.__getitem__(index)
        cell = sample["cell"].squeeze()
        graph = sample["graph"]
        # get data to generate a PyMatGen structure
        atomic_numbers = graph.atomic_numbers
        pos = graph.pos
        struct = make_pymatgen_periodic_structure(
            atomic_numbers, pos, lattice=Lattice(cell)
        )
        comp = struct.composition
        # this gets the composition index, in addition to updating the set
        comp_index = summary.get_composition_index(comp)
        # calculate the average degree of the graph
        avg_degree = degree(graph.edge_index[0]).mean().item()
        # get space group index; spglib breaks for some structures
        # so we just NaN them for now
        try:
            _, spacegroup_number = struct.get_space_group_info()
        except:  # noqa: E722
            space_group_number = np.nan
        # calculate average edge distances (Euclidean based on cart coordinates)
        distances = torch.FloatTensor(
            [
                site.nn_distance
                for dst_site in struct.get_all_neighbors(r=6.0)
                for site in dst_site
            ]
        )
        mean_dist = distances.mean().item()
        std_dist = distances.std().item()
        # package everything into a data structure
        sample_stats = utils.SampleLabels(
            index,
            comp_index,
            comp.reduced_formula,
            graph.num_edges,
            graph.num_nodes,
            avg_degree,
            spacegroup_number,
            mean_dist,
            std_dist,
        )
        # pack the labels as well
        for target_name, value in sample["targets"].items():
            if isinstance(value, torch.Tensor) and len(value) > 1:
                value = value.mean().item()
            setattr(sample_stats, target_name, value)
        summary.sample_labels.append(sample_stats)
    summary_df = summary.to_dataframe()
    # dump the data to a CSV file for analysis
    summary_df.to_csv(f"{key}_statistics.csv", index=False)
    summaries.append(summary)

dump(summaries, "all_summaries.pkl")
