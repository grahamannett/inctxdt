from typing import List, Tuple
import torch
import torch.nn as nn

from inctxdt.config import EnvSpec

from inctxdt.models.layers.base_layers import OriginalEnvEmbedding, OriginalActionHead, BaseInputOutput


def _return_embeds(locs, to_return, embeds, padding_mask):
    return embeds, padding_mask, {k: locs[k] for k in to_return}


class ConvActionDimHead(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        action_dim: int,
        stack_idxs: List[int] = [0, 1],
        kernel_size: int = (1, 2),
        seq_types: int = 3,
    ):
        assert len(stack_idxs) == kernel_size[-1], "we can only stack as many as we can convolve"

        super().__init__()
        self.embedding_dim = embedding_dim
        self.action_dim = action_dim
        self.stack_idxs = stack_idxs
        self.seq_types = seq_types  # i.e. returns, states, actions

        self.norm = nn.LayerNorm(embedding_dim)
        self.conv = nn.Conv2d(embedding_dim, action_dim, kernel_size=kernel_size)
        self.activation = nn.Tanh()

    def forward(self, x: torch.Tensor, **kwargs):
        batch_size, seq_len = x.shape[0], x.shape[1] // self.seq_types

        # go from [batch_size, seq_len * seq_types, emb_dim] -> [batch_size, seq_types, seq_len, emb_dim]
        x = x.reshape(batch_size, seq_len, self.seq_types, self.embedding_dim).permute(0, 2, 1, 3)
        x_postnorm = self.norm(x)

        # stack along this dim realistically we probably dont need this but its confusing otherwise
        # [batch_size, len(stack_idxs), ]
        x_postnorm = torch.stack([x_postnorm[:, i] for i in self.stack_idxs], dim=1)

        # to [batch_size, emb_dim, seq_len, len(stack_idxs)] - treat emb is the channel layer
        x_postnorm = x_postnorm.permute(0, 3, 2, 1)
        x_postnorm = self.conv(x_postnorm)
        x_postnorm = x_postnorm.squeeze(-1).permute(0, 2, 1)  # put back to [batch_size, seq_len, act_dim]

        return self.activation(x_postnorm)

        # return self.activation(x_postnorm).reshape(batch_size, -1)  # flatten to [batch_size, seq_len * act_dim]


class SequentialAction(BaseInputOutput):
    """this module is aimed at taking the input from the original model, and then flattening the actions
    such that they are sequential rather than stacked.  this handles input AND output head"""

    def __init__(
        self,
        embedding_dim: int,
        episode_len: int,
        seq_len: int,
        state_dim: int,
        action_dim: int,
        stack_idxs: List[int] = [0, 1],
        **kwargs,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.action_dim = action_dim
        self.stack_idxs = stack_idxs
        self.seq_types = 1 + 1 + action_dim  # i.e. returns, states, actions

        self.norm = nn.LayerNorm(embedding_dim)
        # self.conv = nn.Conv2d(embedding_dim, action_dim, kernel_size=kernel_size)

        self.episode_len = episode_len
        self.seq_len = seq_len
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.embedding_dim = embedding_dim

        # self.timestep_emb = nn.Embedding(episode_len + seq_len, embedding_dim)
        self.timestep_emb = nn.Embedding(episode_len**2, embedding_dim)
        self.state_emb = nn.Linear(state_dim, embedding_dim)
        self.return_emb = nn.Linear(1, embedding_dim)

        self.action_emb = nn.Linear(1, embedding_dim)  # action MUST be unpacked into time dim
        # self.action_pos_emb = nn.Embedding(action_dim, embedding_dim)

        # output head
        self.action_head = nn.Sequential(nn.Linear(embedding_dim, 1), nn.Tanh())

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
        batch_size, seq_len, action_dim = states.shape[0], states.shape[1], actions.shape[-1]
        spread_dim = 1 + 1 + action_dim  # returns, states, actions_dim

        time_emb = self.timestep_emb(timesteps)

        ret_emb = self.return_emb(returns_to_go.unsqueeze(-1))
        state_emb = self.state_emb(states)

        # note: below is equivalent to self.timestep_emb(timesteps.repeat_interleave(action_dim, -1))
        # which is easier to verify it is correct but less efficient than reusing embedding
        # repeat time emb for actions [batch_size, seq_len, emb_dim, action_dim]
        # repeat_time_emb = time_emb.unsqueeze(-1).repeat_interleave(action_dim, -1)
        # # permute to [batch_size, seq_len, action_dim, emb_dim] -> [batch_size, seq_len * action_dim, emb_dim]
        # repeat_time_emb = repeat_time_emb.permute(0, 1, 3, 2).reshape(
        #     batch_size, -1, self.embedding_dim
        # )  # -- fairly certian the embed is correct for time
        # NOTE: I think when i had the one i thought was equivalent (above) it did not work.  unsure why but maybe need to verify more
        repeat_time_emb = self.timestep_emb(timesteps.repeat_interleave(action_dim, -1))

        # put all actions in the time dim
        # i.e. if we are [batch_size, seq_len, action_dim] -> where without batch it is like:
        # |--------------timestep0--------------|--------------timestep1--------------|
        # [ [ts_0_act0, ts_0_act1, ts_0_act2], [ts1_act1, ts_1_act2, ts_1_act2], ... ]
        # then we can stack them like:
        # |-------------timestep0-------------|-------------timestep1-------------|
        # [ ts_0_act0, ts_0_act1, ts_0_act2, ts_1_act0, ts_1_act1, ts_1_act2, ...]
        # but then this requires having the timesteps repeated like:
        # [ [ts_0, ts_0, ts_0, ts_1, ts_1, ts_1, ...] ]
        # for each batch

        # embed from [batch_size, seq_len, action_dim] -> [batch_size, seq_len * action_dim, 1]
        act_emb = self.action_emb(actions.reshape(batch_size, seq_len * action_dim, 1))

        # add time embed to all
        ret_emb += time_emb
        state_emb += time_emb
        act_emb += repeat_time_emb

        act_emb = act_emb.reshape(batch_size, seq_len, action_dim, self.embedding_dim)
        # seq will be [batch_size, seq_len, (ret) 1 + (state) 1, embedding_dim]

        embeds = torch.stack([ret_emb, state_emb], dim=2)

        # [batch_size, seq_len, (ret) 1 + (state) 1 + (act) action_dim, embedding_dim]
        embeds = torch.cat([embeds, act_emb], dim=2).reshape(batch_size, seq_len * spread_dim, self.embedding_dim)

        if padding_mask is not None:
            # padding mask comes in as [batch_size, seq_len] -> want [batch_size, unpacked_sequence_len]
            padding_mask = torch.stack([padding_mask for _ in range(spread_dim)], dim=-1)
            padding_mask = padding_mask.reshape(batch_size, spread_dim * seq_len)

        # if not return_embs:
        if to_return:
            return _return_embeds(locals(), to_return=to_return, embeds=embeds, padding_mask=padding_mask)
        return embeds, padding_mask

    def forward_output(self, x: torch.Tensor, *args, **kwargs):
        # return self.action_head(x)
        return self.forward_output_linear_action_head(x, *args, **kwargs)

    def forward_output_linear_action_head(self, x: torch.Tensor, *args, **kwargs):
        batch_size = x.shape[0]
        # seq_len = x.shape[1] // self.seq_types

        # reshape to
        x = x.reshape(batch_size, -1, self.seq_types, self.embedding_dim)
        # x = x.reshape(batch_size, seq_len, self.seq_types, self.embedding_dim)
        x = self.norm(x)

        # INFO: not sure if it is going to be [...,:6,:] or [...,-6:,:] i think : self.action_dim is the one that works
        return self.action_head(x[..., : self.action_dim, :]).squeeze(-1).reshape(batch_size, -1)


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
