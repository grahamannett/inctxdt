from typing import List, Optional
import unittest

import torch
from inctxdt.datasets import AcrossEpisodeDataset, MinariDataset, MultipleMinariDataset
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

device = "cuda" if torch.cuda.is_available() else "cpu"


class TestEpisodes(unittest.TestCase):
    def test_dataset(self):
        batch_size = 4
        ds = MinariDataset(env_name="pointmaze-medium-v1")

        dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=Collate(device=device))
        for batch in dl:
            batch = batch.to(device)
            self.assertTrue(str(batch.actions.device) == device)
            break

    def test_combine_dataset(self):
        import time

        dataset_names = ["pointmaze-open-dense-v1", "pointmaze-umaze-v1"]
        datasets = [MinariDataset(env_name=env_name) for env_name in dataset_names]
        dataset = MultipleMinariDataset(datasets)

        time_1 = time.perf_counter()
        dl = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=Collate(device=device))
        for idx, batch in enumerate(dl):
            batch = batch.to(device)
            if idx > 100:
                break
        time_1_end = time.perf_counter()

        time_2 = time.perf_counter()
        dl = torch.utils.data.DataLoader(
            dataset, batch_size=4, shuffle=True, collate_fn=Collate(device=device, return_fn=Batch.with_episode_list)
        )

        for batch in dl:
            batch = batch.to(device)
            if idx > 100:
                break
        time_2_end = time.perf_counter()

        print(f"Time 1: {time_1_end - time_1}")
        print(f"Time 2: {time_2_end - time_2}")

    def test_env(self):
        ds = MinariDataset(env_name="pointmaze-medium-v1")
        env = ds.recover_environment()
        obs = env.reset()
        # breakpoint()

    def test_across_episodes(self):
        ds = AcrossEpisodeDataset(env_name="pointmaze-medium-v1", max_num_epsisodes=3, drop_last=True)
        # dl = torch.utils.data.DataLoader(ds, batch_size=3, shuffle=True, collate_fn=collate_fn)

        out = ds[len(ds) - 2]

        # for batch in dl:
        #     batch.to()
        # # out = next(iter(dl))

        # breakpoint()


if __name__ == "__main__":
    unittest.main()
