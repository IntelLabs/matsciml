from ocpmodels.datasets import transforms as t
from ocpmodels.datasets import IS2REDataset, S2EFDataset, is2re_devset, s2ef_devset


def test_pointcloud_is2re_transform():
    devset = IS2REDataset(is2re_devset, transforms=[t.PointCloudTransform(False)])
    data = next(iter(devset))
    assert all(
        [
            key in data.keys()
            for key in ["atomic_numbers", "pos", "tags", "y_init", "y_relaxed"]
        ]
    )


def test_pointcloud_is2re_shiftcom():
    devset = IS2REDataset(
        is2re_devset, transforms=[t.PointCloudTransform(shift_com=True)]
    )
    data = next(iter(devset))
    assert all(
        [
            key in data.keys()
            for key in ["atomic_numbers", "pos", "tags", "y_init", "y_relaxed"]
        ]
    )


def test_pointcloud_s2ef_transform():
    devset = S2EFDataset(s2ef_devset, transforms=[t.PointCloudTransform(False)])
    data = next(iter(devset))
    assert all(
        [key in data.keys() for key in ["atomic_numbers", "pos", "tags", "y", "force"]]
    )
