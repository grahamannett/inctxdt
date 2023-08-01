import torch
import torch.nn as nn

from inctxdt.episode_data import EpisodeData


class EnvEmbedding(nn.Module):
    def __init__(self, env: "Env", embedding_dim: int, episode_len: int = 1000):
        super().__init__()
        state_dim, action_dim = (
            env.observation_space.shape[0],
            env.action_space.shape[0],
        )

        self.embedding_dict = nn.ModuleDict(
            {
                "timesteps": nn.Embedding(episode_len, embedding_dim),
                "observations": nn.Linear(state_dim, embedding_dim),
                "actions": nn.Linear(action_dim, embedding_dim),
                "rewards": nn.Linear(1, embedding_dim),
            }
        )

    def make_timesteps(self, total_timesteps: int) -> torch.Tensor:
        return torch.arange(total_timesteps)

    def forward(self, samples: EpisodeData) -> torch.Tensor:
        if not hasattr(samples, "timesteps"):
            samples.timesteps = torch.arange(samples.actions.shape[0])

        embs = {
            "observations": self.embedding_dict["observations"](samples.observations),
            "actions": self.embedding_dict["actions"](samples.actions),
            "rewards": self.embedding_dict["rewards"](
                samples.returns_to_go.unsqueeze(-1)
            ),
            "timesteps": self.embedding_dict["timesteps"](samples.timesteps),
        }
        observation_embs = self.embedding_dict["observations"](samples.observations)
        action_embs = self.embedding_dict["actions"](samples.actions)
        reward_embs = self.embedding_dict["rewards"](
            samples.returns_to_go.unsqueeze(-1)
        )
        timestep_embs = self.embedding_dict["timesteps"](samples.timesteps)

        reward_embs += timestep_embs[:, 1:, :]
        action_embs += timestep_embs[:, 1:, :]
        observation_embs += timestep_embs

        embs = torch.stack([reward_embs, observation_embs, action_embs], dim=1)

        # embs = torch.cat(list(embs.values()), dim=1)
        # returns_emb = self.return_emb(returns_to_go.unsqueeze(-1)) + time_emb

        # # [batch_size, seq_len * 3, emb_dim], (r_0, s_0, a_0, r_1, s_1, a_1, ...)
        # sequence = (
        #     torch.stack([returns_emb, state_emb, act_emb], dim=1)
        #     .permute(0, 2, 1, 3)
        #     .reshape(batch_size, 3 * seq_len, self.embedding_dim)
        # )
        return embs


class Model(nn.Module):
    def __init__(
        self,
        env: Env,
        embedding_dim: int = 64,
    ):
        super().__init__()

        # these could be wrapped into one module or moduledict per env
        self.env_embedding = EnvEmbedding(env, embedding_dim=embedding_dim)
        self.head = nn.Linear(embedding_dim, env.action_space.shape[0])

        self.transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=embedding_dim, nhead=4), num_layers=3
        )

    def forward(self, episode):
        emb = self.env_embedding(episode)
        emb = self.transformer(emb, emb)
        emb = self.head(emb)
        return emb
