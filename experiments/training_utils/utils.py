from __future__ import annotations

import os
import socket
import time
import traceback

import pytorch_lightning as pl

pl.seed_everything(6)


def error_log(e, log_path):
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    with open(os.path.join(log_path, "error_log.txt"), "a+") as file:
        file.write("\n" + str(traceback.format_exc()))
        print(traceback.format_exc())


def check_args(args, data_targets):
    for idx, target in enumerate(args.targets):
        data_name = args.data[idx] if len(args.data) > 1 else args.data[0]
        if target not in data_targets[data_name]:
            raise Exception(
                f"Requested target {target} not available in {data_name} dataset.",
                f"Available keys are: {data_targets[data_name]}",
            )


def do_ip_setup():
    ip_addr_file = "./ip_address.txt"

    def get_ip():
        hostname = socket.gethostname()
        ip_addr = socket.gethostbyname(hostname)
        os.environ["MASTER_ADDR"] = ip_addr
        with open(os.path.join(ip_addr_file), "w") as f:
            f.write(ip_addr)

    try:
        os.environ["NODE_RANK"]
    except KeyError:
        os.environ["NODE_RANK"] = "0"

    if os.environ["NODE_RANK"] == "0":
        get_ip()
    else:
        while not os.path.exists(os.path.join(ip_addr_file)):
            time.sleep(1)
        with open(os.path.join(ip_addr_file)) as f:
            master_ip = f.read()
        os.environ["MASTER_ADDR"] = master_ip
