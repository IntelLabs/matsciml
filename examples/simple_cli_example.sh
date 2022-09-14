#!/bin/bash

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: MIT License


# This script demonstrates how to run Open MatSci ML Toolkit using Lightning CLI.
# The three configuration files used conceptually separates data, model, and
# training parameters respectively, and is the recommended way to structure
# your configurations.

# run `python -m ocpmodels.lightning.cli -h` for the help message.
# here, we inform the CLI to run the Lightning "fit" procedure,
# relying on three configuration files
python -m ocpmodels.lightning.cli fit \
	--config ../pl-configs/data/s2ef.yml \
	--config ../pl-configs/models/egnn.yml \
	--config ../pl-configs/trainer.yml
