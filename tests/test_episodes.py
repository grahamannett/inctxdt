import unittest

import torch
from inctxdt.train import MinariDataset, AcrossEpisodeDataset


def collate_fn(batch):
    breakpoint()


datasets = [
    "pointmaze-large-dense-v0",
    "pointmaze-large-v0",
    "pointmaze-medium-dense-v0",
    "pointmaze-medium-v0",
    "pointmaze-open-dense-v0",
    "pointmaze-umaze-v0",
]


class TestEpiodes(unittest.TestCase):
    def test_episodes(self):
        ds = AcrossEpisodeDataset(env_name="pointmaze-medium-v0", max_num_epsisodes=3, drop_last=True)
        dl = torch.utils.data.DataLoader(ds, batch_size=3, shuffle=True, collate_fn=collate_fn)
        # out = next(iter(dl))
        ds_len = len(ds)
        out = ds[len(ds)]

        breakpoint()
