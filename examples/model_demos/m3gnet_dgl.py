from typing import Union, List, Any, Callable, Optional, Dict
from pathlib import Path

import pytorch_lightning as pl
from matgl.ext.pymatgen import Structure2Graph
from matgl.graph.data import M3GNetDataset
from pymatgen.core import Structure

from matsciml.models import M3GNet
from matsciml.models.dgl.m3gnet import element_types
from matsciml.common.registry import registry
from matsciml.datasets.materials_project import MaterialsProjectDataset
from matsciml.lightning.data_utils import MatSciMLDataModule
from matsciml.models.base import ScalarRegressionTask


@registry.register_dataset("M3Dataset")
class M3Dataset(MaterialsProjectDataset):
    def __init__(
        self,
        lmdb_root_path: Union[str, Path],
        threebody_cutoff: float = 4.0,
        cutoff_dist: float = 20.0,
        graph_labels: Union[list[Union[int, float]], None] = None,
        transforms: Optional[List[Callable[..., Any]]] = None,
    ):
        super().__init__(lmdb_root_path, transforms)
        self.threebody_cutoff = threebody_cutoff
        self.graph_labels = graph_labels
        self.cutoff_dist = cutoff_dist

    def _parse_structure(
        self, data: Dict[str, Any], return_dict: Dict[str, Any]
    ) -> None:
        super()._parse_structure(data, return_dict)
        structure: Union[None, Structure] = data.get("structure", None)
        self.structures = [structure]
        self.converter = Structure2Graph(
            element_types=element_types, cutoff=self.cutoff_dist
        )
        graphs, lg, sa = M3GNetDataset.process(self)
        return_dict["graph"] = graphs[0]


# construct a scalar regression task with SchNet encoder
task = ScalarRegressionTask(
    encoder_class=M3GNet,
    encoder_kwargs={
        "element_types": element_types,
    },
    task_keys=["band_gap"],
)

dm = MatSciMLDataModule.from_devset("M3Dataset", num_workers=0, batch_size=4)

# run a quick training loop
trainer = pl.Trainer(fast_dev_run=10)
trainer.fit(task, datamodule=dm)
