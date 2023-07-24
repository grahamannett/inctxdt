from typing import List, Optional
import unittest

import torch
from inctxdt.datasets import AcrossEpisodeDataset, MinariDataset
from inctxdt.batch import Batch, Collate


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


class TestEpisodes(unittest.TestCase):
    def test_dataset(self):
        batch_size = 4
        ds = MinariDataset(env_name="pointmaze-medium-v1")
        device = "cuda" if torch.cuda.is_available() else "cpu"

        dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=Collate(device=device))
        for batch in dl:
            batch = batch.to(device)
            self.assertTrue(batch.actions.device == device)

    def test_across_episodes(self):
        ds = AcrossEpisodeDataset(env_name="pointmaze-medium-v1", max_num_epsisodes=3, drop_last=True)
        # dl = torch.utils.data.DataLoader(ds, batch_size=3, shuffle=True, collate_fn=collate_fn)

        out = ds[len(ds) - 2]

        # for batch in dl:
        #     batch.to()
        # # out = next(iter(dl))

        # breakpoint()
