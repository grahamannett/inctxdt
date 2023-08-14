# import unittest

# import torch
# from inctxdt.d4rl_datasets import D4rlAcrossEpisodeDataset

# from inctxdt.datasets import AcrossEpisodeDataset, MinariDataset, MultipleMinariDataset
# from inctxdt.batch import Collate, return_fn_from_episodes
# from inctxdt.config import config_tool
# from inctxdt.env_helper import get_env


# batch_size = 4
# device = "cuda" if torch.cuda.is_available() else "cpu"
# dataset_names = ["pointmaze-open-dense-v1", "pointmaze-umaze-v1"]


# class TestEpisodes(unittest.TestCase):
#     def test_dataset(self):
#         batch_size = 4
#         ds = MinariDataset(env_name="pointmaze-medium-v1")

#         dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=Collate(device=device))
#         for batch in dl:
#             batch = batch.to(device)
#             self.assertTrue(str(batch.actions.device) == device)
#             break

#     def test_env(self):
#         ds = MinariDataset(env_name="pointmaze-medium-v1")
#         env = ds.recover_environment()
#         obs = env.reset()


# class TestCombinedDataset(unittest.TestCase):
#     def test_default_collate(self):
#         datasets = [MinariDataset(env_name=env_name) for env_name in dataset_names]
#         dataset = MultipleMinariDataset(datasets)
#         dl = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=Collate(device=device))
#         for idx, batch in enumerate(dl):
#             batch = batch.to(device)
#             self.assertTrue(batch.observations.shape[0] == batch_size)
#             if idx > 100:
#                 break

#     def test_alternative_collate(self):
#         datasets = [MinariDataset(env_name=env_name) for env_name in dataset_names]
#         dataset = MultipleMinariDataset(datasets)
#         dl = torch.utils.data.DataLoader(
#             dataset,
#             batch_size=4,
#             shuffle=True,
#             collate_fn=Collate(device=device, return_fn=return_fn_from_episodes(batch_first=True)),
#         )
#         for idx, batch in enumerate(dl):
#             batch = batch.to(device)
#             self.assertTrue(batch.observations.shape[0] == batch_size)
#             if idx > 100:
#                 break


# class TestAcrossEpisodes(unittest.TestCase):
#     def test_across_episodes(self):
#         ds = AcrossEpisodeDataset(env_name="pointmaze-medium-v1", max_num_epsisodes=3)

#         sample = ds[len(ds) - 1]

#         self.assertTrue(sum(sample.total_timesteps) == sample.actions.shape[0])
#         self.assertTrue(sample.id[1] == 0)


# class TestGetEnv(unittest.TestCase):
#     def test_minari_dataset(self):
#         dataset_name = "pointmaze-medium-v1"
#         config = config_tool.get(dataset_name=dataset_name)
#         ds = MinariDataset(dataset_name=config.dataset_name)
#         env_fn, env, venv, obs_space, act_space = get_env(dataset=ds, config=config)

#         new_env = env_fn()
#         obs = new_env.reset()[0]
#         breakpoint()
#         assert obs.shape == obs_space.shape

#     def test_d4rl_dataset(self):
#         dataset_name = "halfcheetah-medium-v2"
#         config = config_tool.get(dataset_name=dataset_name)
#         ds = D4rlAcrossEpisodeDataset(dataset_name=dataset_name, seq_len=config.seq_len)
#         env_fn, env, venv, obs_space, act_space = get_env(dataset=ds, config=config)

#         new_env = env_fn()
#         obs = new_env.reset()
#         assert obs.shape == obs_space.shape


# if __name__ == "__main__":
#     unittest.main()


from tensordict.prototype import tensorclass
import torch


@tensorclass
class MySample:
    x: torch.Tensor
    y: torch.Tensor


@tensorclass
class MyData:
    sample: MySample
    mask: torch.Tensor
    label: torch.Tensor

    def mask_image(self):
        return self.sample.x[self.mask.expand_as(self.sample.x)].view(*self.batch_size, -1)

    def select_label(self, label):
        return self[self.label == label]


x = torch.randn(100, 3, 64, 64)
y = torch.randint(10, (100,))

label = torch.randint(10, (100,))
mask = torch.zeros(1, 64, 64, dtype=torch.bool).bernoulli_().expand(100, 1, 64, 64)


data = MyData(MySample(x, y, batch_size=[100]), mask, label=label, batch_size=[100])

print(data.select_label(1))

print(data.mask_image().shape)

# print(data.reshape(10, 10))
