# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: MIT License
from __future__ import annotations

try:
    import dgl

    _has_dgl = True
except ImportError:
    _has_dgl = False


if _has_dgl:
    pass
