import unittest

import torch
from inctxdt.train import MinariDataset, AcrossEpisodeDataset


def collate_fn(batch):
    breakpoint()


class TestEpiodes(unittest.TestCase):
    def test_episodes(self):
        ds = AcrossEpisodeDataset(env_name="pen-human-v0", max_num_epsisodes=3, drop_last=True)
        dl = torch.utils.data.DataLoader(ds, batch_size=3, shuffle=True, collate_fn=collate_fn)
        # out = next(iter(dl))

        out = ds[20]
        breakpoint()
