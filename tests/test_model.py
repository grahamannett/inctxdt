import unittest

import torch
import torch.nn as nn

from inctxdt.models.model import DecisionTransformer, TransformerBlock


batch_size = 16
state_dim = 17
action_dim = 6
seq_len = 30
device = "cuda"

# Optionally use the context manager to ensure one of the fused kerenels is run


class ELayer(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        # self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(1, embedding_dim)
        self.norm = nn.LayerNorm(embedding_dim)
        self.proj_out = nn.Parameter(torch.rand(embedding_dim, embedding_dim))

    def forward(self, x: torch.Tensor):
        x = x.unsqueeze(-1)
        x = self.linear(x)
        # x = self.dropout(x)
        x = self.norm(x)

        # [batch, seq, obs_dim, proj_dim] x [proj_dim, proj_dim] -> [batch, seq, proj_dim]
        x = torch.einsum("bsop,ed->bsd", x, self.proj_out)
        breakpoint()
        return x


class TestTransformerBlock(unittest.TestCase):
    def test_block(self):
        episode_len = 1000
        embedding_dim = 128
        num_heads = 1
        attention_dropout, residual_dropout = 0.1, 0.1

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(1, 32)
                self.block = TransformerBlock(
                    seq_len=seq_len,
                    embedding_dim=32,
                    num_heads=2,
                    attention_dropout=attention_dropout,
                    residual_dropout=residual_dropout,
                )

            def forward(self, x):
                batch_size, seq_len, state_dim = x.shape

                x = x.unsqueeze(-1)
                x = self.linear(x)
                x = x.reshape(batch_size, -1, 32)
                x = x.permute(0, 2, 1)
                x = self.block(x)
                return x

        model = Model().cuda()
        state_input = torch.rand(batch_size, seq_len, state_dim, device=device)
        out = model(state_input)


class TestModel(unittest.TestCase):
    def test_short_action(self):
        model = DecisionTransformer(state_dim=17, action_dim=6, seq_len=30, episode_len=2048)

        states = torch.rand(batch_size, seq_len, state_dim, device=device)
        actions = torch.rand(batch_size, seq_len, action_dim, device=device)
        mask = torch.ones(batch_size, seq_len, device=device, dtype=torch.int64)
        returns_to_go = torch.rand(batch_size, seq_len, device=device)
        rewards = torch.rand(batch_size, seq_len, device=device)
        timesteps = torch.arange(seq_len, dtype=int, device=device).unsqueeze(0).repeat(batch_size, 1)

        model = model.to(device)
        # out = model(states, actions, returns_to_go, timesteps)

        # # make sure the model works just basic
        # self.assertTrue(out.logits.sum() != 0.0)

        # test if we have taken 1 action so far
        # this actually does not make sense.  if we are looking at the first action of the last timestep we will still have all the actions before that
        out = model(states, actions[:, :, :1], returns_to_go, timesteps)
        self.assertTrue(out.logits.sum() != 0.0)

        # test if we have NO actions
        out = model(states[:, :1, :], returns_to_go=returns_to_go[:, :1], timesteps=timesteps[:, :1], actions=None)
        self.assertTrue(out.logits.sum() != 0.0)
