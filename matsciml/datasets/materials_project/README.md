# Materials Project Database

The [Materials Project Database](https://next-gen.materialsproject.org/) maintained by by Lawrence Berkeley National Laboratory is a database of over 150,000 materials. To use the dataset with Open MatSciML Toolkit the database needs to be downloaded, processed and stored. Alternatively the datasets used by Open MatSciML Toolkit may be downloaded from the Zenodo link.


Setting up MP datasets first requires access to the Materials Project API by creating an account on the [original website](https://materialsproject.org). The API key may then be set to an environment variable: `export MP_API_KEY=your-api-key` to interact with the command-line interface to query for specific data, or rely on pre-configured YAML configurations to process pre-defined splits we refer to in this paper.

Train, validation, and test splits are defined by material id based on the structure-based split, which aims to create a chemically balanced partitioning of the available data. Our dataset splits are informed by the fact that crystal symmetry is a universal property for all of solid-state materials that significantly affects physical properties, including  structure, stability, and functional properties (e.g. band gap, magnetism). In terms of implementation, a simple command line script is used to load material id numbers and download the relevant data to lmdb files, consistent with other datasets used in Open MatSci ML Toolkit. The primary labels used for experiments includes the fields: `band_gap`, `structure`, `formula_pretty`, `efermi`, `symmetry`, `is_metal`, `is_magnetic`, `is_stable`, `formation_energy_per_atom`, `uncorrected_energy_per_atom`, and `energy_per_atom`.


To download and extract the train, validation and test datasets using our code, the following command can be used:

```bash
python -m matsciml.datasets.materials_project.cli \
    -d mp_data \
    -t base \
    -s matsciml/datasets/materials_project/train.yml \
    matsciml/datasets/materials_project/val.yml \
    matsciml/datasets/materials_project/test.yml
```

The `-d` flag is used to specify a directory to store the data, and defaults to `mp_data`. After running the script, the data directory will include train, validation and test folders containing lmdb files with 108159, 30904, and 15,456 samples respectively. Specifying the `-t` flag will ensure all of the main data fields listed above are included in the download.

A devset (development dataset) is also included which has 200 material samples containing the `band_gap`, and `structure` fields, which is accessible in `matsciml/datasets/materials_project/devset`.

Other property fields, material id’s, and Materials Project’s API arguments may be used with the download script to create custom datasets. Additional details on how to use the script may be found in `matsciml/datasets/materials_project/cli.py`.


If you use the Materials Project Database, please refer to [this link](https://next-gen.materialsproject.org/about/cite) for how to cite their work.
