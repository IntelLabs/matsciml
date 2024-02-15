# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: MIT License
from __future__ import annotations

import importlib

try:
    if importlib.util.find_spec("dgl") is not None:
        _has_dgl = True

except ImportError:
    _has_dgl = False


if _has_dgl:
    pass
