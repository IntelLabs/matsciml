
from ocpmodels.datasets.symmetry import symmetry_devset
from ocpmodels.datasets.symmetry.dataset import PointGroupDataset


def test_devset_init():
    dset = PointGroupDataset(symmetry_devset)
    sample = dset.__getitem__(0)
    assert all([key in sample for key in ["pos", "symmetry", "atomic_numbers"]])
    num_particles = sample["pos"].shape[0]
    assert len(sample["atomic_numbers"]) == num_particles


def test_devset_collate():
    dset = PointGroupDataset(symmetry_devset)
    samples = [dset.__getitem__(i) for i in range(8)]
    batch = dset.collate_fn(samples)
