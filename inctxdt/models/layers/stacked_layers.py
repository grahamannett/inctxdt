from typing import List, Tuple

import torch
import torch.nn as nn

from inctxdt.config import EnvSpec, ModalEmbedConfig
from inctxdt.models.layers.base_layers import BaseInputOutput, OriginalActionHead, OriginalEnvEmbedding


class BaseModalEmbedding(nn.Module):
    def _add_time_emb(self, time_emb: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return time_emb


class BaseActionEmbedding(BaseModalEmbedding):
    def __init__(self, action_dim: int, embedding_dim: int, *args, **kwargs):
        super().__init__()
        self.action_dim = action_dim
        self.embedding_dim = embedding_dim
        self._action_head = nn.Sequential(nn.Linear(self.embedding_dim, self.action_dim), nn.Tanh())

    def action_head(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self._action_head(x[:, :, 1, :])

    def combine_embeds(self, *embeds, **kwargs) -> torch.Tensor:
        return torch.stack([*embeds], dim=2)


class ActionEmbedding(BaseActionEmbedding):
    def __init__(self, action_dim: int, embedding_dim: int, *args, **kwargs):
        super().__init__(action_dim=action_dim, embedding_dim=embedding_dim, *args, **kwargs)
        self.action_dim = action_dim
        self.embedding_dim = embedding_dim
        self.action_emb = nn.Linear(action_dim, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.action_emb(x)


class ActionTokenizedEmbedding(BaseActionEmbedding):
    def __init__(self, action_dim: int, embedding_dim: int, token_size: int, pool_fn: str = torch.sum, *args, **kwargs):
        super().__init__(action_dim=action_dim, embedding_dim=embedding_dim, *args, **kwargs)
        self.token_size = token_size
        self.embedding_dim = embedding_dim
        self.pool_fn = pool_fn
        self.action_emb = nn.Embedding(token_size, embedding_dim)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        x = self.action_emb(x)
        return self.pool_fn(x, dim=-2)


class ActionTokenizedSpreadEmbedding(BaseActionEmbedding):
    """
    action comes in as: [bs, seq_len, action_dim] -> [bs, seq_len * action_dim, 1],
    after embedding is:
        [bs, seq_len * action_dim, emb_dim]

    while timesteps are [bs, seq_len] -> [bs, seq_len, emb_dim]
    means we need to repeat the time emb for each action such that it is [bs, seq_len * action_dim, emb_dim]

    to check this you can verify by:
        reshape the timestep emb from [bs, seq_len * action_dim, emb_dim] -> [bs, seq_len, action_dim, emb_dim]
        then check that [:, 0, 0, :] == [:, 0, 1, :] == [:, 0, 2, :] etc
    """

    def __init__(
        self, action_dim: int, embedding_dim: int, token_size: int, max_num_actions: int = 100, *args, **kwargs
    ):
        super().__init__(action_dim=action_dim, embedding_dim=embedding_dim, *args, **kwargs)
        self.action_dim = action_dim
        self.embedding_dim = embedding_dim
        self.token_size = token_size
        self.max_num_actions = max_num_actions

        self.action_emb = nn.Embedding(token_size, embedding_dim)
        self._action_head = nn.Sequential(nn.Linear(self.embedding_dim, 1), nn.Tanh())
        self.action_pos_emb = nn.Parameter(torch.rand(self.max_num_actions, self.embedding_dim))

    def _add_time_emb(self, time_emb: torch.Tensor, tokens: torch.Tensor, **kwargs) -> torch.Tensor:
        # NOTE: cant decide if i should use self.action_dim or pass x in
        # repeat time emb along action dim so that it is [bs, seq_len, action_dim, emb_dim]
        return time_emb.unsqueeze(-2).repeat_interleave(tokens.shape[-1], dim=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # put all actions in the time dim
        # i.e. if we are [bs, seq_len, action_dim] -> where without batch it is like:
        # |--------------timestep0--------------|--------------timestep1--------------|
        # [ [ts_0_act0, ts_0_act1, ts_0_act2], [ts1_act1, ts_1_act2, ts_1_act2], ... ]
        # then we can stack them like:
        # |-------------timestep0-------------|-------------timestep1-------------|
        # [ ts_0_act0, ts_0_act1, ts_0_act2, ts_1_act0, ts_1_act1, ts_1_act2, ...]
        # but then this requires having the timesteps repeated like:
        # [ [ts_0, ts_0, ts_0, ts_1, ts_1, ts_1, ...] ]

        # TODO: fix time embedding, this isnt correct but just refactoring
        # not storing n_steps, action_dim (ie x.shape[1], x.shape[2]) because want to avoid updating those values but it might be quicker than passing in a
        x = self.action_emb(x)

        # add the action pos emb before spreading out
        x += self.action_pos_emb[: x.shape[-2], :]
        return x

    def action_head(self, x: torch.Tensor, **kwargs):
        return self._action_head(x[:, :, 1:-1, :]).squeeze(-1)

    def combine_embeds(
        self, ret_emb: torch.Tensor, state_emb: torch.Tensor, act_emb: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        # possibly you could do this without spreading the act_emb
        #  such that you just cat them BUT! IT PROBABLY WILL MESS UP THE TIME DIM
        # not sure which is quicker between what is below and -> torch.cat([torch.stack([ret_emb, state_emb], dim=2), act_emb], dim=2)
        embeds = torch.stack([ret_emb, state_emb] + [act_emb[..., i, :] for i in range(act_emb.shape[-2])], dim=2)
        return embeds


ModalEmbCls = {
    "ActionEmbedding": ActionEmbedding,
    "ActionTokenizedEmbedding": ActionTokenizedEmbedding,
    "ActionTokenizedSpreadEmbedding": ActionTokenizedSpreadEmbedding,
}


class SequentialAction(BaseInputOutput):
    """this module is aimed at taking the input from the original model, and then flattening the actions
    such that they are sequential rather than stacked.  this handles input AND output head"""

    def __init__(
        self,
        modal_embed_config: ModalEmbedConfig,
        embedding_dim: int,
        episode_len: int,
        seq_len: int,
        state_dim: int,
        action_dim: int,
        **kwargs,
    ):
        super().__init__()
        self.modal_embed_config = modal_embed_config
        self.embedding_dim = embedding_dim
        self.action_dim = action_dim
        self.seq_types = 1 + 1 + action_dim  # i.e. returns, states, actions

        self.episode_len = episode_len
        self.seq_len = seq_len
        self.state_dim = state_dim
        self.action_dim = action_dim

        # EACH MODALITY WILL GET A BRANCH TO EMBED IT
        self.timestep_branch = nn.Embedding(episode_len * 2, self.embedding_dim)
        self.state_branch = nn.Linear(state_dim, self.embedding_dim)
        self.returns_branch = nn.Linear(1, self.embedding_dim)

        # THIS IS HOW YOU WOULD MAKE ONE OF THEM DYNAMIC
        self.action_branch = ModalEmbCls[self.modal_embed_config.action_embed_class](
            action_dim=action_dim, embedding_dim=self.embedding_dim, **modal_embed_config.__dict__
        )

    def forward_embed(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        returns_to_go: torch.Tensor,
        timesteps: torch.Tensor,
        to_return: list[str] = None,
        **kwargs,
    ):
        bs, seq_len = states.shape[0], states.shape[1]
        self._batch_seq_len = seq_len

        time_emb = self.timestep_branch(timesteps)

        ret_emb = self.returns_branch(returns_to_go.unsqueeze(-1))
        state_emb = self.state_branch(states)
        act_emb = self.action_branch(actions)

        # add time emb, since time emb might be need to be repeated, we need to repeat it for each action
        ret_emb += time_emb
        state_emb += time_emb

        act_emb += self.action_branch._add_time_emb(time_emb, actions)

        # this will stack the embeds such that they are [bs, seq_len, seq_types, embedding_dim]
        # note: if you concat along the dim that you create from stack, it will mess up the time dim
        embeds = self.action_branch.combine_embeds(ret_emb, state_emb, act_emb)
        embeds = embeds.reshape(bs, -1, self.embedding_dim)

        self.spread_dim = embeds.shape[1] // seq_len

        return embeds

    def forward_output(self, x: torch.Tensor, *args, **kwargs):
        """last layer of model, or what we learn to predict"""
        # bs = x.shape[0]
        # we want to reshape to reshape to [bs, seq_len, spread_dim, embedding_dim] because
        # the first value of each timestep corresponds to the state dim, regardless of how many actions
        # x = x.reshape(bs, self._batch_seq_len, -1, self.embedding_dim)

        # 1 refers to state -> predict action
        return self.action_branch.action_head(x)
