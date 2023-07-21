import pytest
import os

from ocpmodels.datasets.carolina_db import (
    CMDRequest)


def test_download_data():
    cmd = CMDRequest()
    cmd.material_ids = [0, 1, 2]
    cmd.data_dir = "test/raw_data"
    request_status = cmd.cmd_request()
    assert all(request_status.values())



def test_process_data():
    cmd = CMDRequest()
    cmd.data_dir = "test/raw_data"
    data = cmd.process_data()
    assert all(entry is not None for entry in data)

