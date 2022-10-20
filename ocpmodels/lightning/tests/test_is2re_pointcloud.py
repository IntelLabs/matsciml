from ocpmodels.lightning.data_utils import PointCloudDataModule, IS2REDataset
from ocpmodels.datasets import is2re_devset


def test_grab_batch():
    module = PointCloudDataModule(is2re_devset, IS2REDataset, batch_size=8)
    module.setup()
    loader = module.train_dataloader()
    batch = next(iter(loader))
    # test some position shapes
    pos = batch.get("pos")
    assert pos.ndim == 4
    assert pos.size(0) == 8
    assert pos.size(-1) == 3
    # check feature dimensionality too
    feat = batch.get("pc_features")
    assert feat.ndim == 4
    assert feat.size(0) == 8
    assert feat.size(-1) == 200
