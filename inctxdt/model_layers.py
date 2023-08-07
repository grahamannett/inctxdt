from typing import List, Tuple
import torch
import torch.nn as nn

from inctxdt.config import EnvSpec


# BASE IS WHAT IS MOST SIMILAR TO ORIGINAL DT
class BaseEmbedding(nn.Module):
    def __init__(
        self, episode_len: int, seq_len: int, state_dim: int, action_dim: int, embedding_dim: int = 128, *args, **kwargs
    ) -> None:
        super().__init__()
        self.episode_len = episode_len
        self.seq_len = seq_len

        self.embedding_dim = embedding_dim
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.timestep_emb = nn.Embedding(episode_len + seq_len, embedding_dim)
        self.state_emb = nn.Linear(state_dim, embedding_dim)
        self.action_emb = nn.Linear(action_dim, embedding_dim)
        self.return_emb = nn.Linear(1, embedding_dim)

    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        returns_to_go: torch.Tensor,
        time_steps: torch.Tensor,
        padding_mask: torch.Tensor,
        **kwargs,
    ):
        batch_size, seq_len = states.shape[0], states.shape[1]
        # [batch_size, seq_len, emb_dim]

        time_emb = self.timestep_emb(time_steps)
        state_emb = self.state_emb(states) + time_emb
        act_emb = self.action_emb(actions) + time_emb
        ret_emb = self.return_emb(returns_to_go.unsqueeze(-1)) + time_emb

        # [batch_size, seq_len * 3, emb_dim], (r_0, s_0, a_0, r_1, s_1, a_1, ...)
        sequence = torch.stack([ret_emb, state_emb, act_emb], dim=2).reshape(
            batch_size, 3 * seq_len, self.embedding_dim
        )

        if padding_mask is not None:
            # [batch_size, seq_len * 3], stack mask identically to fit the sequence
            padding_mask = torch.stack([padding_mask, padding_mask, padding_mask], dim=2).reshape(
                batch_size, 3 * seq_len
            )
        return sequence, padding_mask


class DynamicEmbedding(nn.Module):
    def __init__(
        self,
        env_spec: EnvSpec,
        episode_len: int = None,
        seq_len: int = None,
        # state_dim: int = None,
        # action_dim: int = None,
        embedding_dim: int = 128,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim or env_spec.embedding_dim
        self.seq_len = seq_len or env_spec.seq_len

        state_dim = env_spec.state_dim
        action_dim = env_spec.action_dim

        self.env_specs = {env_spec.env_name: env_spec}
        self.env_embeddings = nn.ModuleDict(
            {
                env_spec.env_name: BaseEmbedding(episode_len, seq_len, state_dim, action_dim, embedding_dim),
            }
        )

        self.base_env_name = env_spec.env_name
        self.base_emb = self.env_embeddings[self.base_env_name]

    def setup_base(self):
        """setup base is used when we need to get the clusters

        makes the clusters
        """
        state_weights = self.env_embeddings[self.base_env_name].state_emb.weight
        action_weights = self.env_embeddings[self.base_env_name].action_emb.weight
        return_weights = self.env_embeddings[self.base_env_name].return_emb.weight

    # def unsupervised_train(self, env_emb: "BaseEmbedding"):
    #     state_weights = self.state_emb.weight
    #     action_weights = self.action_emb.weight
    #     return_weights = self.return_emb.weight

    #     kmeans = KMeans(n_clusters=8, mode="euclidean", verbose=1)

    def new_embedding_layer(
        self,
        name: str,
    ):
        env_spec = EnvSpec(episode_len=self.episode_len, seq_len=self.seq_len, env_name=name)
        new_embedding_layer = BaseEmbedding(
            episode_len=self.episode_len,
            seq_len=self.seq_len,
            action_dim=env_spec.action_dim,
            state_dim=env_spec.state_dim,
            embedding_dim=self.embedding_dim,
        )
        new_embedding_layer.unsupervised_train(self.env_specs[self.base_env])
        self.env_embeddings[name] = new_embedding_layer

    def forward(self, env_name: str, *args, **kwargs):
        if env_name not in self.env_embeddings:
            self.new_embedding_layer(env_name)

        return self.env_embeddings[env_name](*args, **kwargs)


class BaseActionHead(nn.Module):
    def __init__(self, action_dim: int, embedding_dim: int = 128, *args, **kwargs) -> None:
        super().__init__()
        self.action_head = nn.Sequential(nn.Linear(embedding_dim, action_dim), nn.Tanh())
        self.max_action = 1.0

    def forward(self, x: torch.Tensor, *args, **kwargs):
        return self.action_head(x[:, 1::3]) * self.max_action


class BaseEmbeddingOutput(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.embed_input = BaseEmbedding(*args, **kwargs)
        self.action_head = BaseActionHead(*args, **kwargs)

    def patch_parent(self, parent):
        parent.embed_input = self.embed_input
        parent.action_head = self.action_head


class StackedEnvEmbedding(nn.Module):
    """this with BaseActionHead should be equivalent to"""

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
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        # self.episode_len = episode_len
        # self.seq_len = seq_len
        # self.state_dim = state_dim
        # self.action_dim = action_dim
        # self.embedding_dim = embedding_dim

        # self.timestep_emb = nn.Embedding(episode_len + seq_len, embedding_dim)
        # self.state_emb = nn.Linear(state_dim, embedding_dim)
        # self.action_emb = nn.Linear(action_dim, embedding_dim)
        # self.return_emb = nn.Linear(1, embedding_dim)

        self.embedding_layer = BaseEmbedding(
            episode_len=episode_len,
            seq_len=seq_len,
            state_dim=state_dim,
            action_dim=action_dim,
            embedding_dim=embedding_dim,
        )
        # self.embedding_layer = DynamicEmbedding()

        if output_head == "default":
            self.action_head = BaseActionHead(action_dim=action_dim, embedding_dim=embedding_dim)
        elif output_head == "sequential":
            self.action_head = StackedActionDimHead(
                embedding_dim, action_dim, stack_idxs=stack_idxs, kernel_size=kernel_size
            )

    # def forward_emb(
    #     self,
    #     states: torch.Tensor,
    #     actions: torch.Tensor,
    #     returns_to_go: torch.Tensor,
    #     time_steps: torch.Tensor,
    #     padding_mask: torch.Tensor,
    #     **kwargs,
    # ):
    #     batch_size, seq_len = states.shape[0], states.shape[1]

    #     time_emb = self.timestep_emb(time_steps)
    #     state_emb = self.state_emb(states) + time_emb
    #     act_emb = self.action_emb(actions) + time_emb
    #     ret_emb = self.return_emb(returns_to_go.unsqueeze(-1)) + time_emb

    #     # [batch_size, seq_len * 3, emb_dim], (r_0, s_0, a_0, r_1, s_1, a_1, ...)
    #     sequence = torch.stack([ret_emb, state_emb, act_emb], dim=2)
    #     sequence = sequence.reshape(batch_size, 3 * seq_len, self.embedding_dim)

    #     if padding_mask is not None:
    #         # [batch_size, seq_len * 3], stack mask identically to fit the sequence
    #         padding_mask = torch.stack([padding_mask, padding_mask, padding_mask], dim=2).reshape(
    #             batch_size, 3 * seq_len
    #         )
    #     return sequence, padding_mask
    def forward_emb(self, *args, **kwargs):
        return self.embedding_layer(*args, **kwargs)

    def forward_output(self, x: torch.Tensor, *args, **kwargs):
        return self.action_head(x)

    def patch_parent(self, parent):
        parent.embed_input = self.forward_emb
        parent.action_head = self.forward_output


class StackedActionDimHead(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        action_dim: int,
        stack_idxs: List[int] = [0, 1],
        kernel_size: int = (1, 2),
    ):
        assert len(stack_idxs) == kernel_size[-1], "we can only stack as many as we can convolve"

        super().__init__()
        self.embedding_dim = embedding_dim
        self.action_dim = action_dim
        self.stack_idxs = stack_idxs
        self.seq_types = 3  # i.e. returns, states, actions

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

        return self.activation(x_postnorm).reshape(batch_size, -1)  # flatten to [batch_size, seq_len * act_dim]


class SequentialAction(nn.Module):
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
        kernel_size: int = (1, 2),
        **kwargs,
    ):
        assert len(stack_idxs) == kernel_size[-1], "we can only stack as many as we can convolve"

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

        self.timestep_emb = nn.Embedding(episode_len + seq_len, embedding_dim)
        self.state_emb = nn.Linear(state_dim, embedding_dim)
        self.return_emb = nn.Linear(1, embedding_dim)

        self.single_action_emb = nn.Linear(1, embedding_dim)
        self.action_pos_emb = nn.Embedding(action_dim, embedding_dim)

        # output head
        self.action_head = nn.Sequential(nn.Linear(embedding_dim, 1), nn.Tanh())

    def forward_emb(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        returns_to_go: torch.Tensor,
        time_steps: torch.Tensor,
        padding_mask: torch.Tensor,
        **kwargs,
    ):
        batch_size, seq_len, action_dim = states.shape[0], states.shape[1], actions.shape[-1]
        spread_dim = 1 + 1 + action_dim  # returns, states, actions_dim

        time_emb = self.timestep_emb(time_steps)

        ret_emb = self.return_emb(returns_to_go.unsqueeze(-1)) + time_emb
        state_emb = self.state_emb(states) + time_emb

        # note: below is equivalent to self.timestep_emb(time_steps.repeat_interleave(action_dim, -1))
        # which is easier to verify it is correct but less efficient than reusing embedding
        # repeat time emb for actions [batch_size, seq_len, emb_dim, action_dim]
        # repeat_time_emb = time_emb.unsqueeze(-1).repeat_interleave(action_dim, -1)
        # # permute to [batch_size, seq_len, action_dim, emb_dim] -> [batch_size, seq_len * action_dim, emb_dim]
        # repeat_time_emb = repeat_time_emb.permute(0, 1, 3, 2).reshape(
        #     batch_size, -1, self.embedding_dim
        # )  # -- fairly certian the embed is correct for time
        repeat_time_emb = self.timestep_emb(time_steps.repeat_interleave(action_dim, -1))

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

        ret_emb += time_emb
        state_emb += time_emb

        # embed from [batch_size, seq_len, action_dim] -> [batch_size, seq_len * action_dim, 1]
        act_emb = self.single_action_emb(actions.reshape(batch_size, seq_len * action_dim, 1))
        pos_act_emb = self.action_pos_emb(torch.arange(action_dim, device=actions.device)).repeat(seq_len, 1)
        act_emb += repeat_time_emb + pos_act_emb

        act_emb = act_emb.reshape(batch_size, seq_len, action_dim, self.embedding_dim)
        # seq will be [batch_size, seq_len, (ret) 1 + (state) 1, embedding_dim]

        embeds = torch.stack([ret_emb, state_emb], dim=2)

        # [batch_size, seq_len, (ret) 1 + (state) 1 + (act) action_dim, embedding_dim]
        embeds = torch.cat([embeds, act_emb], dim=2).reshape(batch_size, seq_len * spread_dim, self.embedding_dim)
        # embeds = embeds.reshape(batch_size, spread_dim * seq_len, self.embedding_dim)

        if padding_mask is not None:
            # TODO: I think i can just stack on dim=2 and then reshape to [batch_size, seq_len * spread_dim]
            padding_mask = torch.stack([padding_mask for _ in range(spread_dim)], dim=1)
            padding_mask = padding_mask.permute(0, 2, 1).reshape(batch_size, -1)

        return embeds, padding_mask

    def forward_output(self, x: torch.Tensor, *args, **kwargs):
        batch_size, seq_len = x.shape[0], x.shape[1] // self.seq_types

        # reshape to
        x = x.reshape(batch_size, seq_len, self.seq_types, self.embedding_dim)
        x = self.norm(x)

        # INFO: not sure if it is going to be [...,:6,:] or [...,-6:,:] i think : self.action_dim is the one that works
        return self.action_head(x[..., : self.action_dim, :]).squeeze(-1).reshape(batch_size, -1)

    # def forward(self, *args, **kwargs):
    #     if "states" in kwargs:
    #         return self.forward_emb(*args, **kwargs)
    #     return self.forward_output(*args, **kwargs)

    def patch_parent(self, parent):
        parent.embed_input = self.forward_emb
        parent.action_head = self.forward_output
