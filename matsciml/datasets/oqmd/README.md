# Open Quantum Materials Database (OQMD)

The [Open Quantum Materials Database](https://oqmd.org/) database of DFT calculated thermodynamic and structural properties of 1,022,603 materials. The data may be accessed in multiple ways, including ad RestAPI and a direct MySQL database download. In the Open MatSciML Toolkit, the RestAPI is used to query and download data, which is then converted to a compatible format to be used with the toolkit.

The endpoint `http://oqmd.org/oqmdapi/formationenergy?&limit={}&offset={}` is used to query all properties of all materials by iteratively supplying an offset and limit. To download specific samples, their offset must be known and a limt of 1 must be used. The available data keys returned are:

- name
- entry_id
- calculation_id
- icsd_id
- formationenergy_id
- duplicate_entry_id
- composition
- composition_generic
- prototype
- spacegroup
- volume
- ntypes
- natoms
- unit_cell
- sites
- band_gap
- delta_e
- stability
- fit
- calculation_label
- atomic_numbers
- cart_coords


and their descriptions may be found [here](https://static.oqmd.org/static/docs/restful.html#:~:text=Available%20keywords%20for%20fields%20and%20filter%C2%B6)

A simple python script may be used to query, save and process the data into an lmdb file:

```python
from matsciml.datasets.oqmd import OQMDRequest

OQMDRequest.make_devset()
oqmd = OQMDRequest(
    base_data_dir="./matsciml/datasets/oqmd"
)
oqmd.download_data()
oqmd.process_json()
oqmd.to_lmdb(oqmd.data_dir)
```

The raw data takes a while to download, mainly limited by the OQMD API itself. Once finished, json files with the offset included in their names will be saved to the `base_data_dir`. An LMDB file will also be generated and saved to the same location.

The ID's may the be used to query the data associated with each material:
```python
from matsciml.datasets.oqmd import OQMDRequest

oqmd = OQMDRequest(split_files=["./base/all.yml"])
oqmd.download_data()
```

Finally, material ID's may be specified specifically along with a `split_dir` to download to:

```python
oqmd = OQMDRequest(
    base_data_dir="./matsciml/datasets/oqmd",
    limit=1,
    split_dir="mini_split",
    material_ids=[1, 2, 3, 4, 5],
)
oqmd.download_data()
oqmd.process_json()
oqmd.to_lmdb(oqmd.data_dir)
```
where in this example, LMDB files would be saved to `./base/mini_split`.

If you use OQMD, please refer to [this link](https://oqmd.org/documentation/publications) for how to cite the work.
