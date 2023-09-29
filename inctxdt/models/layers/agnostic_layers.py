from typing import List, Tuple

import torch
import torch.nn as nn

from inctxdt.config import EnvSpec, ModalEmbedConfig
from inctxdt.models.layers.base_layers import BaseInputOutput, OriginalActionHead, OriginalEnvEmbedding


class StackedEnvEmbedding(BaseInputOutput):
    """this with BaseActionHead should be equivalent to base..."""

    def __init__(
        self,
        episode_len: int,
        seq_len: int,
        state_dim: int,
        action_dim: int,
        embedding_dim: int = 128,
        output_head: str = "default",
        stack_idxs: Tuple[int] = [0, 1],
        kernel_size: Tuple[int] = (1, 2),
        env_spec: EnvSpec = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()

        if state_dim is None or action_dim is None:
            state_dim, action_dim = env_spec.state_dim, env_spec.action_dim

        self.forward_embed = OriginalEnvEmbedding(
            episode_len=episode_len,
            seq_len=seq_len,
            state_dim=state_dim,
            action_dim=action_dim,
            embedding_dim=embedding_dim,
        )

        if output_head == "default":
            self.forward_output = OriginalActionHead(action_dim=action_dim, embedding_dim=embedding_dim)
        elif output_head == "sequential":
            self.forward_output = ConvActionDimHead(
                embedding_dim, action_dim, stack_idxs=stack_idxs, kernel_size=kernel_size
            )
        else:
            raise ValueError(f"output head {output_head} not supported")


class AgnosticEmbed(BaseInputOutput):
    def __init__(
        self,
        action_dim: int,
        embedding_dim: int,
        episode_len: int = 1000,
        **kwargs,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.embedding_dim = embedding_dim
        self.episode_len = episode_len

        self.embed_state = nn.Linear(1, embedding_dim)
        self.emed_act = nn.Linear(1, embedding_dim)
        self.emb_ret = nn.Linear(1, embedding_dim)

        self.timestep_emb = nn.Embedding(episode_len, embedding_dim)

        self.action_head = nn.Sequential(
            nn.Linear(embedding_dim, action_dim),
            nn.Tanh(),
        )

    def forward_embed(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        returns_to_go: torch.Tensor,
        timesteps: torch.Tensor,
        # **kwargs
    ):
        batch_size, seq_len = states.shape[0], states.shape[1]
        state_dim, act_dim = states.shape[-1], actions.shape[-1]

        # unfold the states/etc into the time dim
        states = states.reshape(batch_size, -1, 1)
        actions = actions.reshape(batch_size, -1, 1)
        returns_to_go = returns_to_go.reshape(batch_size, -1, 1)

        state_emb = self.embed_state(states)
        act_emb = self.emed_act(actions)
        ret_emb = self.emb_ret(returns_to_go)

        state_emb += self.timestep_emb(timesteps.repeat_interleave(state_dim, -1))
        act_emb += self.timestep_emb(timesteps.repeat_interleave(act_dim, -1))
        ret_emb += self.timestep_emb(timesteps.repeat_interleave(1, -1))

        ret_emb = torch.nn.functional.adaptive_avg_pool2d(ret_emb, (seq_len, self.embedding_dim))
        state_emb = torch.nn.functional.adaptive_avg_pool2d(state_emb, (seq_len, self.embedding_dim))
        act_emb = torch.nn.functional.adaptive_avg_pool2d(act_emb, (seq_len, self.embedding_dim))

        embeds = torch.stack([ret_emb, state_emb, act_emb], dim=2)
        embeds = embeds.reshape(batch_size, -1, self.embedding_dim)

        return embeds

    def forward_output(self, x: torch.Tensor, *args, **kwargs):
        return self.action_head(x[:, 1::3])
