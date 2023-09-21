from typing import List, Tuple

import torch
import torch.nn as nn


from inctxdt.config import EnvSpec, ModalEmbedConfig
from inctxdt.models.layers.base_layers import BaseInputOutput, OriginalActionHead, OriginalEnvEmbedding
from inctxdt.models.layers.conv_embed_layers import ConvActionDimHead


def _return_embeds(locs, to_return, embeds, padding_mask):
    return embeds, padding_mask, {k: locs[k] for k in to_return}


class BaseModalEmbedding(nn.Module):
    def _add_time_emb(self, time_emb: torch.Tensor) -> torch.Tensor:
        return time_emb


class BaseActionEmbedding(BaseModalEmbedding):
    def __init__(self, action_dim: int, embedding_dim: int, *args, **kwargs):
        super().__init__()
        self.action_dim = action_dim
        self.embedding_dim = embedding_dim
        self._action_head = nn.Sequential(nn.Linear(self.embedding_dim, self.action_dim), nn.Tanh())

    def action_head(self, hs: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self._action_head(hs[:, :, 1, :])


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

    def __init__(self, action_dim: int, embedding_dim: int, token_size: int, *args, **kwargs):
        super().__init__(action_dim=action_dim, embedding_dim=embedding_dim, *args, **kwargs)
        self.action_dim = action_dim
        self.embedding_dim = embedding_dim
        self.token_size = token_size

        # self.action_emb = nn.Embedding(token_size, embedding_dim)
        self.action_emb = nn.Linear(1, embedding_dim)
        self._action_head = nn.Sequential(nn.Linear(self.embedding_dim, 1), nn.Tanh())
        # self.action_pos_emb = nn.Embedding(action_dim**2, embedding_dim)

        self.register_buffer("action_idxs", torch.arange(action_dim**2))

    def _add_time_emb(self, time_emb: torch.Tensor) -> torch.Tensor:
        # time emb coming in will be [bs, seq_len, emb_dim]
        return time_emb.repeat_interleave(self.action_dim, dim=1)

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
        # for each batch

        # TODO: fix time embedding, this isnt correct but just refactoring
        # bs, self.action_n_steps, self.action_dim = x.shape[0], x.shape[1], x.shape[2]
        # x = self.action_emb(x) + self.action_pos_emb(self.action_idxs[: self.action_dim])
        # x = x.reshape(bs, -1, self.embedding_dim)
        bs, self.action_n_steps, self.action_dim = x.shape[0], x.shape[1], x.shape[2]
        x = x.reshape(bs, -1, 1)
        x = self.action_emb(x)
        return x

    def action_head(self, x: torch.Tensor, **kwargs):
        return self._action_head(x[:, :, 1:-1, :]).squeeze(-1)


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

        self.out_norm = nn.LayerNorm(self.embedding_dim)
        # self.conv = nn.Conv2d(embedding_dim, action_dim, kernel_size=kernel_size)

        # self.timestep_emb = nn.Embedding(episode_len + seq_len, embedding_dim)
        self.timestep_emb = nn.Embedding(episode_len * 2, self.embedding_dim)
        self.state_emb = nn.Linear(state_dim, self.embedding_dim)

        self.return_emb = nn.Linear(1, self.embedding_dim)

        ActEmb = ModalEmbCls[self.modal_embed_config.action_embed_class]

        self.action_emb = ActEmb(action_dim=action_dim, embedding_dim=self.embedding_dim, **modal_embed_config.__dict__)

        # output head
        self.observation_head = nn.Sequential(nn.Linear(self.embedding_dim, state_dim), nn.Tanh())

    def forward_embed(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        returns_to_go: torch.Tensor,
        timesteps: torch.Tensor,
        padding_mask: torch.Tensor,
        to_return: list[str] = None,
        **kwargs,
    ):
        bs, seq_len = states.shape[0], states.shape[1]
        self._batch_seq_len = seq_len

        time_emb = self.timestep_emb(timesteps)

        ret_emb = self.return_emb(returns_to_go.unsqueeze(-1))
        state_emb = self.state_emb(states)

        # since we are flattening the actions for some embeddings, we need to repeat the time embedding for each action in some cases
        act_emb = self.action_emb(actions)

        # add time emb, since time emb might be need to be repeated, we need to repeat it for each action
        ret_emb += time_emb
        state_emb += time_emb

        act_emb += self.action_emb._add_time_emb(time_emb)

        # stack embeddings so that we have [bs, seq_len * spread_dim, embedding_dim]
        embeds = torch.cat([ret_emb, state_emb, act_emb], dim=1)
        self.spread_dim = embeds.shape[1] // seq_len

        if padding_mask is not None:
            # padding mask comes in as [bs, seq_len] -> want [bs, unpacked_sequence_len]
            padding_mask = torch.stack([padding_mask for _ in range(self.spread_dim)], dim=-1).reshape(bs, -1)

        return embeds, padding_mask

    def forward_output(self, x: torch.Tensor, *args, **kwargs):
        """last layer of model, or what we learn to predict"""
        bs = x.shape[0]
        # we want to reshape to reshape to [bs, seq_len, spread_dim, embedding_dim] because
        # the first value of each timestep corresponds to the state dim, regardless of how many actions
        x = self.out_norm(x)
        x = x.reshape(bs, self._batch_seq_len, -1, self.embedding_dim)

        # 1 refers to state -> predict action
        act_logits = self.action_emb.action_head(x)
        obs_logits = self.observation_head(x[:, :, 2, :])

        return act_logits, obs_logits


# KEEP THESE FOR POSTERITY UNTIL CERTAIN ITS WORKING
def _old_forward_output_linear_action_head(self, x: torch.Tensor, *args, **kwargs):
    batch_size = x.shape[0]
    x = x.reshape(batch_size, -1, self.seq_types, self.embedding_dim)
    x = self.norm(x)
    x = x[..., 1:, :]
    if x.shape[-2] > 1:
        x = x[..., :-1, :]
    return self.action_head(x).view(batch_size, -1)


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
        padding_mask: torch.Tensor,
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

        if padding_mask is not None:
            # padding mask comes in as [batch_size, seq_len] -> want [batch_size, unpacked_sequence_len]
            padding_mask = torch.stack([padding_mask for _ in range(3)], dim=-1)
            padding_mask = padding_mask.reshape(batch_size, 3 * seq_len)

        return embeds, padding_mask

    def forward_output(self, x: torch.Tensor, *args, **kwargs):
        return self.action_head(x[:, 1::3])
