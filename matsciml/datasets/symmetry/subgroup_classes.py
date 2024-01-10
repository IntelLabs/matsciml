from __future__ import annotations

import collections
import functools

import numpy as np

from matsciml.datasets.symmetry.point_groups import PointGroup, filter_discrete

"""
Original implementation by Matthew Spellings (Vector Institute) 5/25/2023

Modifications by Kelvin Lee to integrate into matsciml
"""


class SubgroupClassMap:
    """Generate a list of point-group symmetries and their subgroups.

    This class generates the group-subgroup relationships within a
    specified set of point groups.

    :param n_max: Maximum symmetry degree to consider
    :param blacklist: List of point groups to block from consideration. Can be individual point group names ('C5', 'D2h') or classes ('polyhedral', 'trivial').

    """

    AXIAL_NAMES = [
        n
        for n in PointGroup._REGISTERED_GROUPS
        if any(n.startswith(prefix) for prefix in ["C", "S", "D"])
    ]
    POLYHEDRAL_NAMES = [
        n
        for n in PointGroup._REGISTERED_GROUPS
        if any(n.startswith(prefix) for prefix in ["T", "O", "I"])
    ]

    def name_expansion(self, name):
        result = []
        if name == "axial":
            for name in self.AXIAL_NAMES:
                result.extend(self.name_expansion(name))
        elif name == "polyhedral":
            for name in self.POLYHEDRAL_NAMES:
                result.extend(self.name_expansion(name))
        elif name == "trivial":
            return ["C1"]
        elif name == "redundant":
            return [
                "C1v",
                "C1h",  #'Cs'
                "D1",  #'C2'
                "S2",  #'Ci'
            ]
        elif name == "Cn":
            return [f"C{i}" for i in range(1, self.n_max + 1)]
        elif name == "Sn" or name == "S2n":
            return [f"S{2 * i}" for i in range(1, self.n_max // 2 + 1)]
        elif name == "Cnh":
            return [f"C{i}h" for i in range(1, self.n_max + 1)]
        elif name == "Cnv":
            return [f"C{i}v" for i in range(1, self.n_max + 1)]
        elif name == "Dn":
            return [f"D{i}" for i in range(1, self.n_max + 1)]
        elif name == "Dnd":
            return [f"D{i}d" for i in range(1, self.n_max + 1)]
        elif name == "Dnh":
            return [f"D{i}h" for i in range(1, self.n_max + 1)]
        else:
            result.append(name)
        return result

    @classmethod
    def update_subgroups(cls, subgroups, processed=None, focus=None):
        processed = processed or set()

        pending = list(subgroups) if focus is None else [focus]
        while pending:
            to_process = pending.pop()
            if to_process in processed:
                continue
            processed.add(to_process)
            for child in list(subgroups[to_process]):
                cls.update_subgroups(subgroups, processed, child)
                subgroups[to_process].update(subgroups[child])

    def __init__(self, n_max=6, blacklist=["trivial", "redundant"]):
        self.n_max = n_max
        self.blacklist = blacklist

        self.full_blacklist = set()
        for name in blacklist:
            self.full_blacklist.update(self.name_expansion(name))

        column_names = set()
        for group in PointGroup._REGISTERED_GROUPS:
            for name in [
                n for n in self.name_expansion(group) if n not in self.full_blacklist
            ]:
                column_names.add(name)
        self.column_names = sorted_column_names = list(sorted(column_names))
        self.column_name_map = {name: i for (i, name) in enumerate(sorted_column_names)}

        self.subgroups = subgroups = collections.defaultdict(lambda: {"C1"})
        for name in sorted_column_names:
            subgroups[name].add(name)

        equiv_groups = [
            ["Ci", "S2"],
            ["C1h", "C1v", "Cs"],
            ["D1", "C2"],
            ["D1h", "C2v"],
            ["D1d", "C2h"],
        ]
        for equiv in equiv_groups:
            for name in equiv:
                subgroups[name].update(equiv)

        subgroups["D2"].add("C2")
        subgroups["D2h"].add("C2")

        for n in range(1, n_max + 1):
            child = f"C{n}"
            parents = [
                name.format(n) for name in ["C{}h", "C{}v", "D{}", "D{}h", "D{}d"]
            ]
            for parent in parents:
                subgroups[parent].add(child)

            names = ["C{}", "C{}h", "C{}v", "D{}", "D{}h", "D{}d"]
            for name in names:
                for mult in range(2, n_max + 1):
                    if n * mult > self.n_max:
                        break
                    subgroups[name.format(mult * n)].add(name.format(n))

                    base_name = name[:3]
                    subgroups[name.format(mult * n)].add(base_name.format(n))

            subgroups[f"C{n}h"].add("Cs")
            subgroups[f"C{n}v"].add("Cs")
            subgroups[f"D{n}h"].add("Cs")
            subgroups[f"D{n}h"].add(f"C{n}h")
            subgroups[f"D{n}h"].add(f"C{n}v")
            subgroups[f"D{n}h"].add(f"D{n}")
            subgroups[f"D{n}d"].add(f"S{2 * n}")
            subgroups[f"D{n}d"].add(f"C{n}v")
            subgroups[f"D{2 * n}h"].add(f"D{n}d")
            subgroups[f"S{2 * n}"].add(f"C{n}")

        # n even
        for n in range(2, n_max + 1, 2):
            for mult in range(1, n_max + 1):
                if n * mult > self.n_max:
                    break
                subgroups[f"S{n * mult}"].add(f"S{n}")
            subgroups[f"C{n}h"].add(f"S{n}")

        # n odd
        for n in range(1, n_max + 1, 2):
            subgroups[f"D{n}d"].add(f"D{n}")

        polyhedral_subgroups = {
            "T": ["C3", "C2", "D2"],
            "Td": ["C3", "S4", "Cs", "T", "D2"],
            "Th": ["S6", "C2", "Cs", "T", "D2", "D2h"],
            "O": ["C2", "C3", "C4", "D2", "T", "D4"],
            "Oh": ["O", "Cs", "O", "D2", "D2h", "C4h", "C4v", "D4h", "D2d", "Td", "Th"],
            "I": ["D3", "D5", "T", "D2"],
            "Ih": ["C3", "C5", "Th", "D2", "I"],
        }
        for name, subs in polyhedral_subgroups.items():
            for sub in subs:
                subgroups[name].add(sub)

        self.update_subgroups(subgroups)

        subgroup_rows = []
        for name in sorted_column_names:
            row = np.zeros(len(sorted_column_names), dtype=np.int32)
            for j, subname in enumerate(sorted_column_names):
                row[j] = subname in subgroups[name]
            subgroup_rows.append(row)
        self.subgroup_rows = np.array(subgroup_rows)
        self.subgroup_row_map = dict(zip(sorted_column_names, self.subgroup_rows))


class SubgroupGenerator:
    """Generate point clouds for training classifiers on point-group symmetries.

    Each element is generated by randomly generating a small cloud of a few points,
    which is then replicated according to the operations of a randomly-selected
    point-group symmetry. Point clouds that exceed a given size are discarded.

    :param n_max: Maximum size of randomly-generated point cloud to be replicated by a selected symmetry operation
    :param sym_max: Maximum symmetry degree to produce
    :param type_max: Maximum number of types to use for point clouds
    :param max_size: Maximum allowed size of replicated point clouds
    :param batch_size: Number of point clouds to generate in each batch
    :param upsample: If True, randomly fill leftover space (up to max_size) in point clouds with identical replicas of points
    :param encoding_filter: Lengthscale to use in `filter_discrete`
    :param blacklist: List of symmetries or symmetry groups to exclude from consideration
    :param multilabel: If True, learn group-subgroup relations in a binary classification setting; if False, learn a single-class classification task
    :param normalize: If True, normalize points to the surface of a sphere; if False, points are allowed to have arbitrary length
    :param lengthscale: Lengthscale for generated point clouds

    """

    BatchType = collections.namedtuple(
        "BatchType",
        [
            "coordinates",
            "source_types",
            "dest_types",
            "label",
            "num_tiles",
            "point_group",
        ],
    )

    def __init__(
        self,
        n_max=4,
        sym_max=6,
        type_max=4,
        max_size=32,
        batch_size=16,
        upsample=False,
        encoding_filter=1e-2,
        blacklist=["trivial", "redundant"],
        multilabel=False,
        normalize=False,
        lengthscale=1.0,
    ):
        self.n_max = n_max
        self.sym_max = sym_max
        self.type_max = type_max
        self.encoding_filter = encoding_filter
        self.max_size = max_size
        self.batch_size = batch_size
        self.upsample = upsample
        self.multilabel = multilabel
        self.normalize = normalize
        self.lengthscale = lengthscale

        self.subgroup_transform_getter = functools.lru_cache(PointGroup.get)

        self.subgroup_map = SubgroupClassMap(sym_max, blacklist)

    def generate(self, seed):
        rng = np.random.default_rng(seed)
        classes = np.array(self.subgroup_map.column_names)
        y_dim = len(self.subgroup_map.column_names)
        orders = {}

        while True:
            batch_r = np.zeros((self.batch_size, self.max_size, 3))
            batch_v = np.zeros((self.batch_size, self.max_size), dtype=np.int64)
            if self.multilabel:
                batch_y = np.zeros((self.batch_size, y_dim), dtype=np.int64)
            else:
                batch_y = np.zeros(self.batch_size, dtype=np.int64)
            i = 0
            while i < self.batch_size:
                name_choice = rng.choice(classes)
                n = min(self.n_max, self.max_size // orders.get(name_choice, 1))
                n = rng.integers(1, max(n, 1), endpoint=True)
                # skip atom number zero to denote padding
                v = rng.integers(1, self.type_max, n)
                r = rng.normal(size=(n, 3))
                symop = self.subgroup_transform_getter(name_choice)
                r = symop(r)
                num_tiles = len(r) // len(v)
                v = np.tile(v, num_tiles)
                r, v = filter_discrete(r, v, self.encoding_filter)
                orders[name_choice] = len(r) / n
                if len(r) <= self.max_size:
                    batch_r[i, : len(r)] = r
                    batch_v[i, : len(v)] = v[:, 0]

                    if self.upsample and len(r) != self.max_size:
                        delta = self.max_size - len(r)
                        indices = rng.integers(0, len(r), delta)
                        batch_r[i, len(r) :] = r[indices]
                        batch_v[i, len(r) :] = v[indices, 0]
                    if self.multilabel:
                        batch_y[i] = self.subgroup_map.subgroup_row_map[name_choice]
                    else:
                        batch_y[i] = self.subgroup_map.column_name_map[name_choice] + 1
                    i += 1

            if self.normalize:
                batch_r /= np.clip(
                    np.linalg.norm(batch_r, axis=-1, keepdims=True),
                    1e-7,
                    np.inf,
                )
            batch_r *= self.lengthscale

            batch_source_v = rng.integers(
                0,
                self.type_max,
                (self.batch_size, self.max_size),
                dtype=np.int64,
            )

            yield self.BatchType(
                batch_r,
                batch_source_v,
                batch_v,
                batch_y,
                num_tiles,
                name_choice,
            )
