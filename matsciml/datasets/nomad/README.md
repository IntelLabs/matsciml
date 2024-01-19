# Novel Materials Discovery (NOMAD)

The [NOMAD database](https://nomad-lab.eu/nomad-lab/index.html) is a freely accessible database of 12,539,160 entries across 2,978,492 materials. Various API requests may be used to query and download many types of materials, properties, and simulation results. To organized a dataset to be used with the Open MatSciML toolkit, two API pipelines need to be use, namely: material ID gathering, and material archive downloading.

A default query for gathering material ID's is provided by the `NomadRequest` module which ensures materials with certain properties are collected:

```json
id_query = {
    "query": {
        "quantities:all": [
            "results.properties.structures",
            "results.properties.structures.structure_original.lattice_parameters",
            "results.properties.structures.structure_original.cartesian_site_positions",
            "results.properties.electronic.dos_electronic",
            "results.properties.electronic.band_structure_electronic",
            "results.material.symmetry",
            "run.calculation.energy",
        ]
    },
    "pagination": {
        "page_size": 10000,
        "order_by": "upload_create_time",
        "order": "desc",
        "page_after_value": "1689167405976:vwonU6jzh1uruW0S9r9Q_JKz1-O0",
        "next_page_after_value": "1689167405976:vwonU6jzh1uruW0S9r9Q_JKz1-O0",
    },
    "required": {"include": ["entry_id"]},
}
```

The [Nomad Explore](https://nomad-lab.eu/prod/v1/gui/search/entries) page was used to craft the query, which is helpful for understanding how many materials match the requested query. In the above case, 139,039 entries should be available.

The material ID's may then be queried and saved with a simple script:

```python
from matsciml.datasets.nomad import NomadRequest

nomad = NomadRequest(base_data_dir="./base")
nomad.fetch_ids()
```

The ID's will be saved to the `base_data_dir` in `.yml` format.

The ID's may the be used to query the data associated with each material:
```python
from matsciml.datasets.nomad import NomadRequest

nomad = NomadRequest(split_files=["./base/all.yml"])
nomad.download_data()
```

An LMDB file will be saved for each split file in a directory: `./base_dir/split_name`.


Finally, material ID's may be specified specifically along with a `split_dir` to download to:

```python
from matsciml.datasets.nomad import NomadRequest

nomad = NomadRequest(
    base_data_dir="./base",
    split_dir="mini_split",
    material_ids={
        0: "GjAKByPxraKfkFCdFrwp0omDVQZ7",
        1: "0FwC9lqZWvGigWMtxgdn7M6YXhwu",
        2: "VSRNiGFB2epCnn6OBY04S4175SIY",
        3: "wvfvLz6S0xj7S8oXVIpEbDdh1hwD",
        4: "OldNS7xP3AtG_NT3uFEyrlk1xh20",
    },
)
nomad.download_data()
```

where in this example, LMDB files would be saved to `./base/mini_split`.

If you use NOMAD, please refer to [this link](https://nomad-lab.eu/nomad-lab/cite.html) for how to cite the work.
