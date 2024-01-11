# Carolina Materials Database

The [Carolina Materials Database](http://www.carolinamatdb.org/) from University of South Carolina is a freely, globally accessible database of 214,436 inorganic material compounds with over 250,000 calculated properties. To use the dataset with Open MatSciML Toolkit the database needs to be downloaded, processed and stored.

To download the full dataset, the `CMDRequest` module from `matsciml/datasets/carolina_db/carolina_api.py` may be used as follows:

```python
from matsciml.datasets.carolina_db import CMDRequest

cmd = CMDRequest(base_data_dir="./matsciml/datasets/carolina_db/base")
cmd.download_data()
cmd.process_data()

```

By default, all available data will be downloaded in `.cif` format. Note that two endpoints are used to download the data - one for a main file at: "http://www.carolinamatdb.org/static/export/cifs/{}.cif" which is the main `.cif` file, as well as "http://www.carolinamatdb.org/entry/{}/" which is used to scrape the Formation Energy property from it's entries webpage, which is not included in the main `.cif`. The energy is appended to the main file, and saved in a `raw_data` directory. Once all files have been downloaded, the data is processed with relevant properties extracted and saved into a python dictionary, which is then serialized into `lmdb`format. Some entries may fail to download due to overloading the remote server with requests. If this happens, failed sample ID's will be collected in `./{base_data_dir}/raw_data/failed.txt`. The failed file list is removed every time `CDBRequest.cmd_request()` is called.


If samples fail to download, the best option to retrieve them would be to specify their material id's and the split they belong to, for example:

```python
from matsciml.datasets.carolina_db import CMDRequest

cmd = CMDRequest(
    base_data_dir="./matsciml/datasets/carolina_db/base",
    split_dir="all",
    material_ids=[
        123123,
    ],
)
cmd.download_data()
```


Similar to the above snippet, sample id's to download may also be specified by a list of material id's, or from a split file. Split files are yaml files named for the split the represent, and with newline separated material id's inside. Predefined split files for train, test, and validation are provided.

```python
from matsciml.datasets.carolina_db import CMDRequest

# Specify material ID's
# defaults to 'all' split folder, unless otherwise specified by `split_dir`
cmd = CMDRequest(
    base_data_dir="./matsciml/datasets/carolina_db/base",
    material_ids=[0, 1, 2],
    split_dir="all'
)
# OR specify split files
cmd = CMDRequest(
    base_data_dir="./matsciml/datasets/carolina_db/base",
    split_files=["train.yml", "test.yml", "val.yml"],
)
cmd.download_data()
cmd.process_data()

```

If you use Carolina Material Database, please refer to [this link](http://www.carolinamatdb.org/docs/) for how to cite their work
