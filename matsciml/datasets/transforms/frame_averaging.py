# Frame Averaging implementation from: https://github.com/vict0rsch/faenet/tree/main
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import math
import numbers
import random
from copy import deepcopy
from itertools import product

import torch


from matsciml.common import package_registry
from matsciml.datasets.transforms.base import AbstractDataTransform

if package_registry["dgl"]:
    import dgl

if package_registry["pyg"]:
    import torch_geometric
    from torch_geometric.transforms import LinearTransformation

def compute_frames(
    eigenvec,
    pos,
    cell,
    fa_method="stochastic",
    pos_3D=None,
    det_index=0,
):
    """Compute all `frames` for a given graph, i.e. all possible
    canonical representations of the 3D graph (of all euclidean transformations).

    Args:
        eigenvec (tensor): eigenvectors matrix
        pos (tensor): centered position vector
        cell (tensor): cell direction (dxd)
        fa_method (str): the Frame Averaging (FA) inspired technique
            chosen to select frames: stochastic-FA (stochastic), deterministic-FA (det),
            Full-FA (all) or SE(3)-FA (se3).
        pos_3D: for 2D FA, pass atoms' 3rd position coordinate.

    Returns:
        (list): 3D position tensors of projected representation
    """
    dim = pos.shape[1]  # to differentiate between 2D or 3D case
    plus_minus_list = list(product([1, -1], repeat=dim))
    plus_minus_list = [torch.tensor(x) for x in plus_minus_list]
    all_fa_pos = []
    all_cell = []
    all_rots = []
    assert fa_method in {
        "all",
        "stochastic",
        "det",
        "se3-all",
        "se3-stochastic",
        "se3-det",
    }
    se3 = fa_method in {
        "se3-all",
        "se3-stochastic",
        "se3-det",
    }
    fa_cell = deepcopy(cell)

    if fa_method == "det" or fa_method == "se3-det":
        sum_eigenvec = torch.sum(eigenvec, axis=0)
        plus_minus_list = [torch.where(sum_eigenvec >= 0, 1.0, -1.0)]

    for pm in plus_minus_list:
        # Append new graph positions to list
        new_eigenvec = pm * eigenvec

        # Consider frame if it passes above check
        fa_pos = pos @ new_eigenvec

        if pos_3D is not None:
            full_eigenvec = torch.eye(3)
            fa_pos = torch.cat((fa_pos, pos_3D.unsqueeze(1)), dim=1)
            full_eigenvec[:2, :2] = new_eigenvec
            new_eigenvec = full_eigenvec

        if cell is not None:
            fa_cell = cell @ new_eigenvec

        # Check if determinant is 1 for SE(3) case
        if se3 and not torch.allclose(
            torch.linalg.det(new_eigenvec),
            torch.tensor(1.0),
            atol=1e-03,
        ):
            continue

        all_fa_pos.append(fa_pos)
        all_cell.append(fa_cell)
        all_rots.append(new_eigenvec.unsqueeze(0))

    # Handle rare case where no R is positive orthogonal
    if all_fa_pos == []:
        all_fa_pos.append(fa_pos)
        all_cell.append(fa_cell)
        all_rots.append(new_eigenvec.unsqueeze(0))

    # Return frame(s) depending on method fa_method
    if fa_method == "all" or fa_method == "se3-all":
        return all_fa_pos, all_cell, all_rots

    elif fa_method == "det" or fa_method == "se3-det":
        return [all_fa_pos[det_index]], [all_cell[det_index]], [all_rots[det_index]]

    index = random.randint(0, len(all_fa_pos) - 1)
    return [all_fa_pos[index]], [all_cell[index]], [all_rots[index]]


def check_constraints(eigenval, eigenvec, dim=3):
    """Check that the requirements for frame averaging are satisfied

    Args:
        eigenval (tensor): eigenvalues
        eigenvec (tensor): eigenvectors
        dim (int): 2D or 3D frame averaging
    """
    # Check eigenvalues are different
    if dim == 3:
        if (eigenval[1] / eigenval[0] > 0.90) or (eigenval[2] / eigenval[1] > 0.90):
            print("Eigenvalues are quite similar")
    else:
        if eigenval[1] / eigenval[0] > 0.90:
            print("Eigenvalues are quite similar")

    # Check eigenvectors are orthonormal
    if not torch.allclose(eigenvec @ eigenvec.T, torch.eye(dim), atol=1e-03):
        print("Matrix not orthogonal")

    # Check determinant of eigenvectors is 1
    if not torch.allclose(torch.linalg.det(eigenvec), torch.tensor(1.0), atol=1e-03):
        print("Determinant is not 1")


def frame_averaging_3D(pos, cell=None, fa_method="stochastic", check=False):
    """Computes new positions for the graph atoms using
    frame averaging, which itself builds on the PCA of atom positions.
    Base case for 3D inputs.

    Args:
        pos (tensor): positions of atoms in the graph
        cell (tensor): unit cell of the graph. None if no pbc.
        fa_method (str): FA method used
            (stochastic, det, all, se3-all, se3-det, se3-stochastic)
        check (bool): check if constraints are satisfied. Default: False.

    Returns:
        (tensor): updated atom positions
        (tensor): updated unit cell
        (tensor): the rotation matrix used (PCA)
    """

    # Compute centroid and covariance
    pos = pos - pos.mean(dim=0, keepdim=True)
    C = torch.matmul(pos.t(), pos)

    # Eigendecomposition
    eigenval, eigenvec = torch.linalg.eigh(C)

    # Sort, if necessary
    idx = eigenval.argsort(descending=True)
    eigenvec = eigenvec[:, idx]
    eigenval = eigenval[idx]

    # Check if constraints are satisfied
    if check:
        check_constraints(eigenval, eigenvec, 3)

    # Compute fa_pos
    fa_pos, fa_cell, fa_rot = compute_frames(eigenvec, pos, cell, fa_method)

    # No need to update distances, they are preserved.

    return fa_pos, fa_cell, fa_rot


def frame_averaging_2D(pos, cell=None, fa_method="stochastic", check=False):
    """Computes new positions for the graph atoms using
    frame averaging, which itself builds on the PCA of atom positions.
    2D case: we project the atoms on the plane orthogonal to the z-axis.
    Motivation: sometimes, the z-axis is not the most relevant one (e.g. fixed).

    Args:
        pos (tensor): positions of atoms in the graph
        cell (tensor): unit cell of the graph. None if no pbc.
        fa_method (str): FA method used (stochastic, det, all, se3)
        check (bool): check if constraints are satisfied. Default: False.

    Returns:
        (tensor): updated atom positions
        (tensor): updated unit cell
        (tensor): the rotation matrix used (PCA)
    """

    # Compute centroid and covariance
    pos_2D = pos[:, :2] - pos[:, :2].mean(dim=0, keepdim=True)
    C = torch.matmul(pos_2D.t(), pos_2D)

    # Eigendecomposition
    eigenval, eigenvec = torch.linalg.eigh(C)
    # Sort eigenvalues
    idx = eigenval.argsort(descending=True)
    eigenval = eigenval[idx]
    eigenvec = eigenvec[:, idx]

    # Check if constraints are satisfied
    if check:
        check_constraints(eigenval, eigenvec, 3)

    # Compute all frames
    fa_pos, fa_cell, fa_rot = compute_frames(
        eigenvec,
        pos_2D,
        cell,
        fa_method,
        pos[:, 2],
    )
    # No need to update distances, they are preserved.

    return fa_pos, fa_cell, fa_rot


def data_augmentation(g, d=3, *args):
    """Data augmentation: randomly rotated graphs are added
    in the dataloader transform.

    Args:
        g (data.Data): single graph
        d (int): dimension of the DA rotation (2D around z-axis or 3D)
        rotation (str, optional): around which axis do we rotate it.
            Defaults to 'z'.

    Returns:
        (data.Data): rotated graph
    """

    # Sampling a random rotation within [-180, 180] for all axes.
    if d == 3:
        transform = RandomRotate([-180, 180], [0, 1, 2])  # 3D
    else:
        transform = RandomRotate([-180, 180], [2])  # 2D around z-axis

    # Rotate graph
    graph_rotated, _, _ = transform(g)

    return graph_rotated


class RandomRotate:
    r"""Rotates node positions around a specific axis by a randomly sampled
    factor within a given interval.

    Args:
        degrees (tuple or float): Rotation interval from which the rotation
            angle is sampled. If `degrees` is a number instead of a
            tuple, the interval is given by :math:`[-\mathrm{degrees},
            \mathrm{degrees}]`.
        axes (int, optional): The rotation axes. (default: `[0, 1, 2]`)
    """

    def __init__(self, degrees, axes=[0, 1, 2]):
        if isinstance(degrees, numbers.Number):
            degrees = (-abs(degrees), abs(degrees))
        assert isinstance(degrees, (tuple, list)) and len(degrees) == 2
        self.degrees = degrees
        self.axes = axes

    def __call__(self, data):
        if data.pos.size(-1) == 2:
            degree = math.pi * random.uniform(*self.degrees) / 180.0
            sin, cos = math.sin(degree), math.cos(degree)
            matrix = [[cos, sin], [-sin, cos]]
        else:
            m1, m2, m3 = torch.eye(3), torch.eye(3), torch.eye(3)
            if 0 in self.axes:
                degree = math.pi * random.uniform(*self.degrees) / 180.0
                sin, cos = math.sin(degree), math.cos(degree)
                m1 = torch.tensor([[1, 0, 0], [0, cos, sin], [0, -sin, cos]])
            if 1 in self.axes:
                degree = math.pi * random.uniform(*self.degrees) / 180.0
                sin, cos = math.sin(degree), math.cos(degree)
                m2 = torch.tensor([[cos, 0, -sin], [0, 1, 0], [sin, 0, cos]])
            if 2 in self.axes:
                degree = math.pi * random.uniform(*self.degrees) / 180.0
                sin, cos = math.sin(degree), math.cos(degree)
                m3 = torch.tensor([[cos, sin, 0], [-sin, cos, 0], [0, 0, 1]])

            matrix = torch.mm(torch.mm(m1, m2), m3)

        data_rotated = LinearTransformation(matrix)(data)
        if torch_geometric.__version__.startswith("2."):
            matrix = matrix.T

        # LinearTransformation only rotates `.pos`; need to rotate `.cell` too.
        if hasattr(data_rotated, "cell"):
            data_rotated.cell = torch.matmul(
                data_rotated.cell,
                matrix.to(data_rotated.cell.device),
            )
        # And .force too
        if hasattr(data_rotated, "force"):
            data_rotated.force = torch.matmul(
                data_rotated.force,
                matrix.to(data_rotated.force.device),
            )

        return (
            data_rotated,
            matrix,
            torch.inverse(matrix),
        )

    def __repr__(self):
        return "{}({}, axis={})".format(
            self.__class__.__name__,
            self.degrees,
            self.axis,
        )


class FrameAveraging(AbstractDataTransform):
    r"""Frame Averaging (FA) Transform for (PyG) Data objects (e.g. 3D atomic graphs).

    Computes new atomic positions (`fa_pos`) for all datapoints, as well as
    new unit cells (`fa_cell`) attributes for crystal structures, when applicable.
    The rotation matrix (`fa_rot`) used for the frame averaging is also stored.

    Args:
        frame_averaging (str): Transform method used.
            Can be 2D FA, 3D FA, Data Augmentation or no FA, respectively denoted by
            (`"2D"`, `"3D"`, `"DA"`, `""`)
        fa_method (str): the actual frame averaging technique used.
            "stochastic" refers to sampling one frame at random (at each epoch),
            "det" to chosing deterministically one frame, and "all" to using all frames.
            The prefix "se3-" refers to the SE(3) equivariant version of the method.
            "" means that no frame averaging is used.
            (`""`, `"stochastic"`, `"all"`, `"det"`, `"se3-stochastic"`, `"se3-all"`, `"se3-det"`)

    Returns:
        (data.Data): updated data object with new positions (+ unit cell) attributes
        and the rotation matrices used for the frame averaging transform.
    """

    def __init__(self, frame_averaging=None, fa_method=None):
        self.fa_method = (
            "stochastic" if (fa_method is None or fa_method == "") else fa_method
        )
        self.frame_averaging = "" if frame_averaging is None else frame_averaging
        self.inactive = not self.frame_averaging
        assert self.frame_averaging in {
            "",
            "2D",
            "3D",
            "DA",
        }
        assert self.fa_method in {
            "",
            "stochastic",
            "det",
            "all",
            "se3-stochastic",
            "se3-det",
            "se3-all",
        }

        if self.frame_averaging:
            if self.frame_averaging == "2D":
                self.fa_func = frame_averaging_2D
            elif self.frame_averaging == "3D":
                self.fa_func = frame_averaging_3D
            elif self.frame_averaging == "DA":
                self.fa_func = data_augmentation
            else:
                raise ValueError(f"Unknown frame averaging: {self.frame_averaging}")

    def __call__(self, data):
        """The only requirement for the data is to have a `pos` attribute."""
        graph = data["graph"]
        if self.inactive:
            return data
        elif self.frame_averaging == "DA":
            return self.fa_func(data, self.fa_method)
        else:
            graph.fa_pos, graph.fa_cell, graph.fa_rot = self.fa_func(
                graph.pos,
                data["cell"] if "cell" in data else None,
                self.fa_method,
            )
            data["graph"] = graph
            return data
