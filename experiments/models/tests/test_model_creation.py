from __future__ import annotations

import pytest

from experiments.utils.utils import instantiate_arg_dict

from experiments.models import available_models

models = list(available_models.keys())
models.remove("generic")


@pytest.mark.parametrize("model", models)
def test_instantiate_model_dict(model):
    model_dict = available_models[model]
    instantiate_arg_dict(model_dict)
