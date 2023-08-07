import unittest


import torch

from inctxdt.config import EnvSpec
from inctxdt.d4rl_datasets import D4rlDataset
from inctxdt.model_layers import DynamicEmbedding


class TestEmbeddingLayer(unittest.TestCase):
    def test_base_works(self):
        base_spec = EnvSpec(episode_len=1000, seq_len=200, env_name="halfcheetah-medium-v2")

        ds = D4rlDataset(env_name=base_spec.env_name, seq_len=base_spec.seq_len)
        dl = torch.utils.data.DataLoader(ds, batch_size=32, num_workers=8)
        self.embedding_layer = DynamicEmbedding(env_spec=base_spec)

        batch = next(iter(dl))

        out = self.embedding_layer(batch)
