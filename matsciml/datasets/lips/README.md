# LiPS: Lithium Phosphorus Sulfide

The LiPS data splits used in the experiments are included in the codebase folders `matsciml/datastes/lips/base/{train, val, test}`. To create the splits, we download the dataset from it's [original release](https://archive.materialscloud.org/record/2022.45) and split randomly into 70%, 20% and 10% chunks for training, validation and testing. A dev set is also included in `matsciml/datasets/lips/devset` which holds 200 samples.

The following script may be used to create the random splits:
```python
from tqdm import tqdm
import numpy as np

from matsciml.datasets.lips import LiPSDataset
from matsciml.datasets.lips.parser import LiPSStructure
from matsciml.datasets.base import PointCloudDataset


dataset_path = "PATH_TO_DATASET"

dset = LiPSDataset(dataset_path)
req = LiPSStructure()


data = [None] * len(dset)
for idx in tqdm(range(len(data)), total=len(data), desc="Processed"):
    data[idx] = PointCloudDataset.data_from_key(dset, 0, 1)

req.data = data

split_percents = {"train": 0.7, "val": 0.2, "test": 0.1}

split_percent_values = list(split_percents.values())
indices = {k: [] for k in split_percents.keys()}

split_map = {"random": list(range(len(dset)))}

for prop_key, prop_idx in split_map.items():
    prop_idx = np.random.permutation(prop_idx)
    split_idx = [
        int(sum(split_percent_values[:idx]) * len(prop_idx))
        for idx in range(1, len(split_percent_values))
    ]
    data_splits = np.split(prop_idx, split_idx)
    sub_dict = dict(zip(indices.keys(), data_splits))
    for sub_k, sub_v in sub_dict.items():
        indices[sub_k].extend(list(sub_v))

og_data = req.data.copy()
for split_name, index_list in indices.items():
    req.data_dir = split_name
    id_list = []
    req.data = []
    for idx in index_list:
        req.data.append(og_data[idx])

    req.to_lmdb(req.data_dir)
```

The LiPS dataset contains `energy` and `force` keys which may be used for regression tasks.


If you use the LiPS Database, please refer to [this link](https://archive.materialscloud.org/record/2022.45#:~:text=How%20to%20cite%20this%20record) for how to cite their work.
