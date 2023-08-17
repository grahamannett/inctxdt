import unittest

import torch

from inctxdt.models.model import DecisionTransformer


class TestModel(unittest.TestCase):
    def test_short_action(self):
        batch_size = 16
        state_dim = 17
        action_dim = 6
        seq_len = 30
        device = "cuda"

        model = DecisionTransformer(state_dim=17, action_dim=6, seq_len=30, episode_len=2048)

        states = torch.rand(batch_size, seq_len, state_dim, device=device)
        actions = torch.rand(batch_size, seq_len, action_dim, device=device)
        mask = torch.ones(batch_size, seq_len, device=device, dtype=torch.int64)
        returns_to_go = torch.rand(batch_size, seq_len, device=device)
        rewards = torch.rand(batch_size, seq_len, device=device)
        timesteps = torch.arange(seq_len, dtype=int, device=device).unsqueeze(0).repeat(batch_size, 1)

        model = model.to(device)
        out = model(states, actions, returns_to_go, timesteps)

        # make sure the model works just basic
        self.assertTrue(out.logits.sum() != 0.0)

        # test if we have taken 1 action so far
        out = model(states, actions[:, :, :1], returns_to_go, timesteps)
        self.assertTrue(out.logits.sum() != 0.0)

        # test if we have NO actions
        out = model(states[:, :1, :], returns_to_go=returns_to_go[:, :1], timesteps=timesteps[:, :1], actions=None)
        self.assertTrue(out.logits.sum() != 0.0)
