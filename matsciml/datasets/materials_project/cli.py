from __future__ import annotations

from argparse import ArgumentParser
from json import dump, loads
from pathlib import Path

import yaml

from matsciml.datasets.materials_project import MaterialsProjectRequest

"""
Functions as a simple CLI for querying and saving the Materials Project.

The two forms of interaction with this CLI is either using a preset query,
or by creating your own custom dataset. A minimal example of the former:

> python -m matsciml.datasets.materials_project.cli -t devset

...retrieves a "dev" split, which is really just an extremely small query
and used mainly for testing purposes. Specifying "base" will try to more
exhaustively query the dataset.

For custom usage, please refer to `MaterialsProjectRequest`, as the CLI arguments
map directly onto the class'.
"""

presets = {
    "devset": {
        "fields": ["band_gap", "structure"],
        "api_kwargs": {"num_sites": (2, 100), "num_chunks": 2, "chunk_size": 100},
    },
    "base": {
        "fields": [
            "band_gap",
            "structure",
            "formula_pretty",
            "efermi",
            "symmetry",
            "is_metal",
            "is_magnetic",
            "is_stable",
            "formation_energy_per_atom",
            "uncorrected_energy_per_atom",
            "energy_per_atom",
        ],
        "api_kwargs": {"num_sites": (2, 10000), "num_chunks": 200, "chunk_size": 1000},
    },
}

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-d",
        "--data_dir",
        type=Path,
        help="Root directory that holds Open MatSciML Toolkit data; typically same as OCP.",
    )
    parser.add_argument(
        "-f",
        "--fields",
        nargs="*",
        help="String names of fields to request separated by spaces, e.g. -f band_gap structure.",
        required=False,
    )
    parser.add_argument(
        "-m",
        "--material_ids",
        nargs="*",
        help="String names of materials to request separated by spaces, e.g. -m mp-1071555 mp-1101139.",
        required=False,
    )
    parser.add_argument(
        "-a",
        "--api_key",
        type=str,
        help="Optional API key for Materials Project. If nothing is provided, will parse from MP_API_KEY environment variable.",
        required=False,
    )
    parser.add_argument(
        "-k",
        "--api_kwargs",
        type=loads,
        help="Dictionary key/value pairs to pass into the API query, e.g. -k {'num_chunks': 3, 'num_sites': [5, 10]} . See `MPRester.summary.search` documentation.",
        required=False,
    )
    parser.add_argument(
        "-t",
        "--task",
        type=str,
        choices=list(presets.keys()),
        help="Choose a preset task dataset to download. Mutually exclusive with other arguments.",
        required=False,
    )
    parser.add_argument(
        "-s",
        "--split_files",
        nargs="*",
        help="Files that define splits for train, test and validation sets which are specified by material id., e.g. -s train.yml val.yml.",
        required=False,
    )
    args = vars(parser.parse_args())
    data_dir = args["data_dir"]
    # delete key to prevent unpacking
    del args["data_dir"]
    task = args.get("task", None)
    split_files = args.get("split_files", None)
    # task takes precedent over everything else
    if task is not None:
        parameters = presets.get(task)
        api_kwargs = parameters.get("api_kwargs")
        del parameters["api_kwargs"]
        # update api_kwargs based on user overrides
        user_kwargs = args.get("api_kwargs")
        if user_kwargs is None:
            user_kwargs = {}
        api_kwargs.update(user_kwargs)
        if split_files:
            for file in split_files:
                with open(file) as f:
                    data = yaml.safe_load(f)
                    split, ids = next(iter(data.keys())), next(iter(data.values()))
                client = MaterialsProjectRequest(
                    material_ids=ids,
                    **parameters,
                    **api_kwargs,
                )
                data = client.retrieve_data()
                # if we have a predefined task the name is just the "split"
                name = task
                target = data_dir.joinpath(name, split)
                client.to_lmdb(target)
        else:
            client = MaterialsProjectRequest(**parameters, **api_kwargs)
            data = client.retrieve_data()
            # if we have a predefined task the name is just the "split"
            name = task
    else:
        from hashlib import sha256

        # get rid of unneeded key, and unpack kwargs
        del args["task"]
        api_kwargs = args.get("api_kwargs", {})
        if len(api_kwargs) > 0:
            del args["api_kwargs"]
        # otherwise just build the request from inputs
        client = MaterialsProjectRequest(**args, **api_kwargs)
        data = client.retrieve_data()
        model = sha256()
        for key in ["fields", "material_ids", "api_kwargs"]:
            value = args.get(key, None)
            if value is not None:
                model.update(bytes(str(value), "utf-8"))
        hashkey = model.hexdigest()[:7]
        name = f"custom/{hashkey}"
    # used to name the data split
    target = data_dir.joinpath(name)
    # export data to target folder
    if split_files is None:
        client.to_lmdb(target)
    with open(target.joinpath("request.json"), "w+") as write_file:
        dump(client.to_dict(), write_file)
