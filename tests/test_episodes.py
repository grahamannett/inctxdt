import unittest

import torch


from inctxdt.batch import Collate, return_fn_from_episodes
from inctxdt.config import Config
from inctxdt.d4rl_datasets import D4rlAcrossEpisodeDataset, D4rlDataset
from inctxdt.env_helper import get_env
from inctxdt.minari_datasets import AcrossEpisodeDataset, MinariDataset, MultipleMinariDataset
from inctxdt.datasets_meta import Discretizer

batch_size = 4
device = "cuda" if torch.cuda.is_available() else "cpu"
dataset_names = ["pointmaze-open-dense-v1", "pointmaze-umaze-v1"]


class TestEpisodes(unittest.TestCase):
    def test_dataset(self):
        batch_size = 4
        ds = MinariDataset(env_name="pointmaze-medium-v1")

        dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=Collate(device=device))
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
        dl = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=Collate(device=device))
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
            collate_fn=Collate(device=device, return_fn=return_fn_from_episodes(batch_first=True)),
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


class TestGetEnv(unittest.TestCase):
    def test_minari_dataset(self):
        dataset_name = "pointmaze-medium-v1"
        config = Config(dataset_name=dataset_name)
        ds = MinariDataset(dataset_name=config.env_name)
        env_fn, env, venv, obs_space, act_space = get_env(dataset=ds, config=config)

        new_env = env_fn()
        obs = new_env.reset()[0]
        assert obs.shape == obs_space.shape

    def test_d4rl_dataset(self):
        dataset_name = "halfcheetah-medium-v2"
        config = Config(dataset_name=dataset_name)
        ds = D4rlAcrossEpisodeDataset(dataset_name=dataset_name, seq_len=config.seq_len)
        env_fn, env, venv, obs_space, act_space = get_env(dataset=ds, config=config)

        new_env = env_fn()
        obs = new_env.reset()
        assert obs.shape == obs_space.shape


class TestTokenized(unittest.TestCase):
    def test_tokenized_datset(self):
        dataset_name = "halfcheetah-medium-v2"
        dataset = D4rlDataset(dataset_name=dataset_name, seq_len=10)

        ep0 = dataset[0]

        discretizer = Discretizer()
        dataset = discretizer.use_with_dataset(dataset)

        ep1 = dataset[0]


if __name__ == "__main__":
    unittest.main()
