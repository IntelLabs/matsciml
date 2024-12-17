from __future__ import annotations

import pytest
import numpy as np

from matsciml.datasets.utils import Edge


def test_non_self_interaction():
    """
    These two nodes edges should not be equivalent, since they
    are not self-interactions and the images are different
    """
    a = Edge(src=0, dst=10, image=np.array([-1, 0, 0]))
    b = Edge(src=0, dst=10, image=np.array([1, 0, 0]))
    assert a != b


def test_self_interaction_image():
    """
    These two edges are mirror images of one another since
    the src/dst are the same node.
    """
    a = Edge(src=0, dst=0, image=np.array([-1, 0, 0]))
    b = Edge(src=0, dst=0, image=np.array([1, 0, 0]))
    assert a == b


@pytest.mark.parametrize("is_undirected", [True, False])
def test_directed_edges(is_undirected):
    """
    These two are the same edge in the undirected case,
    but are different if treating directed graphs
    """
    a = Edge(src=5, dst=10, image=np.array([0, 0, 0]), is_undirected=is_undirected)
    b = Edge(src=10, dst=5, image=np.array([0, 0, 0]), is_undirected=is_undirected)
    if is_undirected:
        assert a == b
    else:
        assert a != b
