import unittest


import torch
from inctxdt.batch import Batch, Collate

from inctxdt.config import EnvSpec, Config
from inctxdt.d4rl_datasets import D4rlDataset
from inctxdt.layers.dynamic_layers import DynamicEmbedding, DynamicLayers


device = "cuda"


class TestEmbeddingLayer(unittest.TestCase):
    def test_dynamic_embedding(self):
        # base_spec = EnvSpec(episode_len=1000, seq_len=200, env_name="halfcheetah-medium-v2")

        base_env = "halfcheetah-medium-v2"
        alt_env = "hopper-medium-v0"
        batch_size = 32
        episode_len, seq_len = 1000, 20
        env_spec = Config._get_env_spec(env_name=base_env, episode_len=1000, seq_len=20)
        env_spec2 = Config._get_env_spec(env_name=alt_env, episode_len=1000, seq_len=20)

        embedding_layer = DynamicEmbedding(env_spec=env_spec)
        embedding_layer.cuda()

        ds = D4rlDataset(env_name=base_env, seq_len=20)
        ds2 = D4rlDataset(env_name=alt_env, seq_len=20)

        # breakpoint()

        dl = torch.utils.data.DataLoader(ds, batch_size=32, num_workers=8, collate_fn=Collate())
        dl2 = torch.utils.data.DataLoader(ds2, batch_size=32, num_workers=8, collate_fn=Collate())

        batch = next(iter(dl))
        batch = batch.to(device)

        timesteps = torch.arange(seq_len, device=device).unsqueeze(0).repeat(batch_size, 1)

        out = embedding_layer(
            env_name=batch.env_name[0],
            states=batch.states,
            actions=batch.actions,
            returns_to_go=batch.returns_to_go,
            timesteps=timesteps,
            padding_mask=batch.make_padding_mask(),
        )

        batch2 = next(iter(dl2))
        batch2 = batch2.to(device)

        out = embedding_layer(
            env_name=batch2.env_name[0],
            states=batch2.states,
            actions=batch2.actions,
            returns_to_go=batch2.returns_to_go,
            timesteps=batch2.timesteps,
            padding_mask=batch2.make_padding_mask(),
        )

    def test_with_stubs(self):
        episode_len, seq_len = 1000, 20
        batch_size = 32

        base_env = "halfcheetah-medium-v2"
        env_spec = Config._get_env_spec(env_name=base_env, episode_len=episode_len, seq_len=seq_len)

        def _make_batch_for_env(env_name, state_dim, action_dim):
            batch = Batch(
                states=torch.rand(batch_size, seq_len, state_dim, device=device),
                actions=torch.rand(batch_size, seq_len, action_dim, device=device),
                mask=torch.ones(batch_size, seq_len, device=device, dtype=torch.int64),
                returns_to_go=torch.rand(batch_size, seq_len, device=device),
                rewards=torch.rand(batch_size, seq_len, device=device),
                timesteps=torch.arange(seq_len, dtype=int, device=device).unsqueeze(0).repeat(batch_size, 1),
                # total_timesteps=torch.tensor(),
                env_name=env_name,
                batch_size=[batch_size],
            )
            return batch

        batch = _make_batch_for_env(env_spec.env_name, env_spec.state_dim, env_spec.action_dim)

        embedding_layer = DynamicEmbedding(env_spec=env_spec)
        embedding_layer.cuda()

        out = embedding_layer(
            env_name=batch.env_name,
            states=batch.states,
            actions=batch.actions,
            returns_to_go=batch.returns_to_go,
            timesteps=batch.timesteps,
            padding_mask=batch.make_padding_mask(),
        )

        cluster_info = embedding_layer.make_env_clusters()

        alt_env_spec = Config._get_env_spec(env_name="hopper-medium-v0", episode_len=episode_len, seq_len=seq_len)
        batch2 = _make_batch_for_env(alt_env_spec.env_name, alt_env_spec.state_dim, alt_env_spec.action_dim)

        embedding_out, pad_mask_out = embedding_layer(
            env_name=batch2.env_name,
            states=batch2.states,
            actions=batch2.actions,
            returns_to_go=batch2.returns_to_go,
            timesteps=batch2.timesteps,
            padding_mask=batch2.make_padding_mask(),
        )

        def _get_loss(cent):
            return torch.einsum("bse,ce->be", embedding_out, cent)

        loss = (
            _get_loss(cluster_info["state"].centroids)
            + _get_loss(cluster_info["action"].centroids)
            + _get_loss(cluster_info["return"].centroids)
            # + _get_loss(cluster_info["timestep"].centroids)
        )

        loss = -loss.mean()
        loss.backward()

        dynamic_model = DynamicLayers(env_spec=env_spec)
        dynamic_model.cuda()

        model_output = dynamic_model(
            env_name=batch.env_name,
            states=batch.states,
            actions=batch.actions,
            returns_to_go=batch.returns_to_go,
            timesteps=batch.timesteps,
            padding_mask=batch.make_padding_mask(),
        )

        d_out = dynamic_model(
            env_name=batch2.env_name,
            states=batch2.states,
            actions=batch2.actions,
            returns_to_go=batch2.returns_to_go,
            timesteps=batch2.timesteps,
            padding_mask=batch2.make_padding_mask(),
        )

        # breakpoint()

        # embeds = embedding_out
        # centers = cluster_info["state"].centroids

        # centers_embedding = torch.einsum("bse,ce->be", embeds, centers)
        # loss = -centers_embedding.mean()

        # breakpoint()

        # numerator = torch.exp((embeds.T @ centers) / temp)
        # denominator = torch.exp((embeds.T @ centers) / temp) + torch.sum(torch.exp((embeds.T @ negatives) / temp))

        # return -torch.log(numerator / denominator)
