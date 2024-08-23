import torch
from torch import nn

from matsciml.lightning.callbacks import embedding_magnitude_hook
from matsciml.common.types import Embeddings


class DummyEncoder(nn.Module):
    def forward(self, g_z, n_z) -> Embeddings:
        embeddings = Embeddings(g_z, n_z)
        return embeddings


def test_hook_manual(caplog):
    g_z = torch.rand(8, 64) * 30
    n_z = torch.rand(340, 64) * 30
    encoder = DummyEncoder()
    encoder.register_forward_hook(embedding_magnitude_hook)
    _ = encoder(g_z, n_z)
    assert "WARNING" in caplog.text
    assert "embedding value is greater" in caplog.text
