from typing import List, Optional
import unittest

import torch
import time
from inctxdt.datasets import AcrossEpisodeDataset, MinariDataset, MultipleMinariDataset
from inctxdt.batch import Batch, Collate, return_fn_from_episodes


def collate_fn(batch):
    breakpoint()


batch_size = 4

dataset_names = ["pointmaze-open-dense-v1", "pointmaze-umaze-v1"]

device = "cuda" if torch.cuda.is_available() else "cpu"

# idk if i need this
# print("warming up with random tensor to device")
# data = torch.rand(100, 100, 100)
# data = data.to(device)
# data *= 0.123
# data = data.cpu()
# print("done warmup...")


class TestEpisodes(unittest.TestCase):
    def test_dataset(self):
        batch_size = 4
        ds = MinariDataset(env_name="pointmaze-medium-v1")

        dl = torch.utils.data.DataLoader(
            ds, batch_size=batch_size, shuffle=True, collate_fn=Collate(device=device)
        )
        for batch in dl:
            batch = batch.to(device)
            self.assertTrue(str(batch.actions.device) == device)
            break

    def test_env(self):
        ds = MinariDataset(env_name="pointmaze-medium-v1")
        env = ds.recover_environment()
        obs = env.reset()


class TestCombinedDataset(unittest.TestCase):
    def test_default_collate(self):
        datasets = [MinariDataset(env_name=env_name) for env_name in dataset_names]
        dataset = MultipleMinariDataset(datasets)
        dl = torch.utils.data.DataLoader(
            dataset, batch_size=4, shuffle=True, collate_fn=Collate(device=device)
        )
        for idx, batch in enumerate(dl):
            batch = batch.to(device)
            self.assertTrue(batch.observations.shape[0] == batch_size)
            if idx > 100:
                break

    def test_alternative_collate(self):
        datasets = [MinariDataset(env_name=env_name) for env_name in dataset_names]
        dataset = MultipleMinariDataset(datasets)
        dl = torch.utils.data.DataLoader(
            dataset,
            batch_size=4,
            shuffle=True,
            collate_fn=Collate(
                device=device, return_fn=return_fn_from_episodes(batch_first=True)
            ),
        )
        for idx, batch in enumerate(dl):
            batch = batch.to(device)
            self.assertTrue(batch.observations.shape[0] == batch_size)
            if idx > 100:
                break


class TestAcrossEpisodes(unittest.TestCase):
    def test_across_episodes(self):
        ds = AcrossEpisodeDataset(env_name="pointmaze-medium-v1", max_num_epsisodes=3)

        sample = ds[len(ds) - 1]

        self.assertTrue(sum(sample.total_timesteps) == sample.actions.shape[0])
        self.assertTrue(sample.id[1] == 0)


if __name__ == "__main__":
    unittest.main()
