import warnings
import os
import time
from contextlib import suppress
from functools import cached_property
from typing import Dict, List, Optional, Union

import pandas as pd
import requests
import yaml
from tqdm import tqdm
import lmdb
from ocpmodels.datasets.utils import write_lmdb_data


class CMDRequest:
    def __init__(
        self,
        base_data_dir: str = "./",
        split_files: Optional[List[str]] = None,
        material_ids: Optional[List[int]] = None,
    ):
        self.base_data_dir = base_data_dir
        os.makedirs(base_data_dir, exist_ok=True)
        self.material_ids = material_ids
        self.split_files = split_files
        # Can specify material ids' or split_files, but not both. If neither are
        # supplied, the full dataset is downloaded.
        if material_ids and split_files:
            raise Exception(f"Only one of material_ids or split_files may be supplied")
        elif material_ids is None and split_files is None:
            self.material_ids = list(range(0, 214435))

    @property
    def material_ids(self) -> Union[List[str], None]:
        return self._material_ids

    @material_ids.setter
    def material_ids(self, values: Union[List[int], None]) -> None:
        self._material_ids = values

    @property
    def data_dir(self) -> Union[List[str], None]:
        return self._data_dir

    @data_dir.setter
    def data_dir(self, dst_folder: Union[str, None]) -> None:
        """Use the `base_data_dir` plus the destination folder to create `data_dir`.
        The `dst_folder` is determined by which split we are processing specified by
        `split_files`, otherwise will default to `all`.

        Args:
            dst_folder (Union[str, None]): Destination folder for data (raw, and lmdb)
        """
        self._data_dir = os.path.join(self.base_data_dir, dst_folder)
        os.makedirs(self._data_dir, exist_ok=True)

    def process_ids(self) -> Dict[str, int]:
        """Builds a dictionary of split names and the id's associated with them.
        If not split files are specified then whatever material id's are present are
        used to create the 'all' split.

        Returns:
            Dict[str, int]: _description_
        """
        ids = {}
        if self.split_files is not None:
            for split_file in self.split_files:
                ids[split_file] = yaml.safe_load(open(split_file, "r"))
        else:
            ids["all"] = self.material_ids
        return ids

    def download_data(self):
        """Facilitates downloading data from different splits. Makes a folder
        specifically for the raw .cif files.
        """
        id_dict = self.process_ids()
        for split, ids in id_dict.items():
            self.material_ids = ids
            self.data_dir = os.path.join(os.path.splitext(split)[0], "raw_data")
            self.cmd_request()

    def cmd_request(self) -> None:
        """ "Queries the Carolina Materials Database through a specific endpoint.
        Becasue the database is hosted on a machine with 1 CPU and limited memory, some
        queries may fail. A pause between requests was suggested by the CMD creator.
        If queries fail, a text file with samples that failed will be saved so they can
        be requested again if desired.
        """
        url = "http://www.carolinamatdb.org/static/export/cifs/{}.cif"
        request_status = {}

        with suppress(OSError):
            os.remove(os.path.join(self.data_dir, f"failed.txt"))

        for n in tqdm(
            self.material_ids, desc="Total Processed: ", total=len(self.material_ids)
        ):
            time.sleep(0.001)
            data = requests.get(url=url.format(n))
            if data.status_code == 200:
                data = data.text
                with open(os.path.join(self.data_dir, f"{n}.cif"), "w") as f:
                    f.write(data)
                request_status[n] = True
            else:
                warning_message = f"Sample {n} from {self.data_dir} failed to download with code: {data.status_code}\n"
                warnings.warn("warning_message")
                with open(os.path.join(self.data_dir, f"failed.txt"), "a") as f:
                    f.write(warning_message)
                request_status[n] = False
        return request_status

    @cached_property
    def atomic_number_map(self) -> Dict[str, int]:
        """List of element symbols and their atomic numbers.

        Returns:
            Dict[str, int]: _description_
        """
        return dict(pd.read_csv("atomic_number_map.txt", header=None).values)

    def process_data(self) -> Dict:
        """Processes the raw .cif data. Grabbing any properties or attributes that are
        provides, as well as the position data. Gathers everything into a dictionary,
        and then saves to LMDB at the end.

        Returns:
            Dict: _description_
        """
        files = os.listdir(self.data_dir)
        self.data = [None] * len(files)
        for idx, file in enumerate(files):
            data = open(os.path.join(self.data_dir, file), "r").read()
            data_dict = {}
            # files come with training blank line
            lines = data.split("\n")
            if lines[-1] == "":
                lines = lines[:-1]
            # remove the header lines:
            lines = lines[2:]
            lines = [line.strip() for line in lines]
            # split lines into property lines, and remainder
            property_lines, lines = (
                lines[: lines.index("loop_")],
                lines[lines.index("loop_") + 1 :],
            )
            props = [property.split(maxsplit=1) for property in property_lines]
            # make a property dictionay and update data_dict
            prop_dict = dict(pd.DataFrame(props).values)
            data_dict.update(prop_dict)
            # split next set of lines to get the symmetry info
            symmetry_lines, lines = (
                lines[
                    lines.index("_symmetry_equiv_pos_as_xyz") + 1 : lines.index("loop_")
                ],
                lines[lines.index("loop_") + 1 :],
            )
            symmetries = pd.DataFrame(
                [symmetry.split(maxsplit=1) for symmetry in symmetry_lines]
            )
            symmetries[0] = pd.to_numeric(symmetries[0])
            symmetry_dict = dict(symmetries.values)
            data_dict["symmetry_dict"] = symmetry_dict
            # split remaining lines to get the positions and atomic numbers
            pos_lines = lines[lines.index("_atom_site_occupancy") + 1 :]
            pos_df = pd.DataFrame([p.split() for p in pos_lines])
            pos = pos_df[[3, 4, 5]].to_numpy(dtype=float)
            atomic_numbers = [
                self.atomic_number_map[symbol] for symbol in pos_df[0].values
            ]
            data_dict["atomic_numbers"] = atomic_numbers
            data_dict["pos"] = pos
            self.data[idx] = data_dict

        self.to_lmdb(os.path.dirname(self.data_dir))
        return self.data

    def to_lmdb(self, lmdb_path: str) -> None:
        """
        Save the retrieved documents to an LMDB file.

        Requires specifying a folder to save to, in which a "data.lmdb" file
        and associated lockfile will be created. Each entry is saved as key/
        value pairs, with keys simply being the index, and the value being
        a pickled dictionary representation of the retrieved `SummaryDoc`.

        Parameters
        ----------
        lmdb_path : str
            Directory to save the LMDB data to.

        Raises
        ------
        ValueError:
            [TODO:description]
        """
        os.makedirs(lmdb_path, exist_ok=True)
        target_env = lmdb.open(
            os.path.join(lmdb_path, "data.lmdb"),
            subdir=False,
            map_size=1099511627776 * 2,
            meminit=False,
            map_async=True,
        )
        if self.data is not None:
            for index, entry in tqdm(
                enumerate(self.data), desc="Entries processed", total=len(self.data)
            ):
                write_lmdb_data(index, entry, target_env)
        else:
            raise ValueError(
                f"No data was available for serializing - did you run `retrieve_data`?"
            )

    @classmethod
    def devset(cls):
        kwargs = {
            "base_data_dir": "./devset",
            "material_ids": list(range(0, 100)),
        }
        return cls(**kwargs)
