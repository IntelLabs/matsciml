#!/bin/bash
# This uses the matsciml Lightning CLI wrapper to configure
# the training workflow. The advantage of doing so is the
# ability to modularize experiments, i.e. not have to redefine
# datasets, models, and/or trainer control.
python -m matsciml.lightning.cli fit \
	--config data.yml \
	--config egnn.yml \
	--config trainer.yml \
	--trainer.fast_dev_run 20 # override some config if we need to
