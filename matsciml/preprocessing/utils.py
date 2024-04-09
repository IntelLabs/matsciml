from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from pymatgen.core import Composition

"""
This module houses some utility dataclasses for ease
of abstraction. They are not currently used anywhere
else in the ``matsciml`` workflow except here.
"""


@dataclass
class SampleLabels:
    index: int
    composition_index: int
    reduced_composition: str
    num_edges: int
    num_nodes: int
    avg_degree: float
    spacegroup_number: int
    mean_dist: float
    std_dist: float


@dataclass
class DatasetSummary:
    dataset_name: str
    composition_set: dict[Composition, int]
    sample_labels: list[SampleLabels]

    def get_composition_index(self, composition: Composition) -> int:
        if composition not in self.composition_set:
            self.composition_set[composition] = len(self.composition_set)
        return self.composition_set[composition]

    def to_dataframe(self) -> pd.DataFrame:
        dicts = [sample.__dict__ for sample in self.sample_labels]
        return pd.DataFrame(dicts)
