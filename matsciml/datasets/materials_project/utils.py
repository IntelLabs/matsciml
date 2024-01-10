from __future__ import annotations

from collections import defaultdict

import torch

torch.manual_seed(21510)


def get_split_map(data, property):
    if property == "crystal_class":
        crystal_map = {
            "1": "tri",
            "-1": "tri",
            "2": "mono",
            "m": "mono",
            "2/m": "mono",
            "222": "ortho",
            "mm2": "ortho",
            "mmm": "ortho",
            "4": "tetra",
            "-4": "tetra",
            "4/m": "tetra",
            "422": "tetra",
            "4mm": "tetra",
            "-42m": "tetra",
            "-4m2": "tetra",
            "4/mmm": "tetra",
            "3": "tri",
            "-3": "tri",
            "312": "tri",
            "321": "tri",
            "31m": "tri",
            "3m1": "tri",
            "-31m": "tri",
            "-3m1": "tri",
            "-3m": "tri",
            "32": "tri",
            "3m": "tri",
            "6": "hex",
            "-6": "hex",
            "6/m": "hex",
            "622": "hex",
            "6mm": "hex",
            "-6m2": "hex",
            "-62m": "hex",
            "6/mmm": "hex",
            "23": "cubic",
            "m-3": "cubic",
            "432": "cubic",
            "-43m": "cubic",
            "m-3m": "cubic",
        }

        split_map = defaultdict(list)
        for idx, sample in enumerate(data):
            split_map[crystal_map[sample.symmetry.point_group]].append(idx)
    else:
        raise NotImplementedError

    return split_map
