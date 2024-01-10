from __future__ import annotations

import logging
import zipfile
from argparse import ArgumentParser
from pathlib import Path
from shutil import move, which
from subprocess import PIPE, run

import requests

parser = ArgumentParser()
parser.add_argument(
    "--data-path",
    type=Path,
    default="../data",
    help="Path to move the unpacked data to.",
)
args = parser.parse_args()

logging.basicConfig(level=logging.INFO)
repo_checksum = "e40104f89bef406af90dbbc6355643c2"
logging.debug("Creating request to Zenodo servers")
record = requests.get("https://zenodo.org/api/records/7411133").json()
# retrieve the data with the correct checksum for this repository version
valid = False
for index, entry in enumerate(record["files"]):
    record_checksum = entry["checksum"].replace("md5:", "")
    if record_checksum == repo_checksum:
        valid = True
        break
if not valid:
    raise ValueError(
        f"No match for repository checksum on Zenodo servers. Expected: {repo_checksum}",
    )
target_data = record["files"][index]
# get MD5 checksum for validation
logging.info(f"Repository expected data MD5 hash: {repo_checksum}")
logging.info(f"Zenodo data MD5 hash: {record_checksum}")
target_link = target_data["links"]["self"]
filename = Path(target_link).name
logging.info(f"Downloading data from {target_link}")

# download the data file
assert which("wget") is not None, "wget was not found in your path; please install it!"
logging.info("Downloading dataset from Zenodo")
_ = run(["wget", target_link])
# now compare checksums
logging.info("Running checksum check; this can take a minute.")
checksum_proc = run(f"md5sum < {filename}", stdout=PIPE, shell=True)
dl_checksum = checksum_proc.stdout.decode("utf-8").split()[0]
logging.info(f"Downloaded data MD5 hash: {dl_checksum}")
assert (
    dl_checksum == record_checksum
), f"Checksum mismatch; expected: {checksum}, downloaded: {dl_checksum}. Contact maintainers!"

# unpack the zip file
logging.info("Unpacking dataset zip")
with zipfile.ZipFile(filename, "r") as z:
    z.extractall(".")
move("dgl_is2re/is2re", args.data_path)
