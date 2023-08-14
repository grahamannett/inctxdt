import torch
import torch.nn as nn


class BaseInputOutput(nn.Module):
    def patch_parent(self, parent):
        parent.forward_embed = self.forward_embed
        parent.forward_output = self.forward_output

        return self

    def forward(self, *args, **kwargs):
        if "states" in kwargs:
            return self.forward_emb(*args, **kwargs)
        return self.forward_output(*args, **kwargs)


# BASE IS WHAT IS MOST SIMILAR TO ORIGINAL DT
class OriginalEnvEmbedding(nn.Module):
    def __init__(
        self,
        episode_len: int,
        seq_len: int,
        state_dim: int,
        action_dim: int,
        embedding_dim: int = 128,
        types_in_step: int = 3,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()
        self.episode_len = episode_len
        self.seq_len = seq_len

        self.types_in_step = types_in_step  # 3 for (r, s, a)

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
        timesteps: torch.Tensor,
        padding_mask: torch.Tensor = None,
        **kwargs,
    ):
        batch_size, seq_len = states.shape[0], states.shape[1]
        # [batch_size, seq_len, emb_dim]

        time_emb = self.timestep_emb(timesteps)
        state_emb = self.state_emb(states) + time_emb
        act_emb = self.action_emb(actions) + time_emb
        ret_emb = self.return_emb(returns_to_go.unsqueeze(-1)) + time_emb

        # [batch_size, seq_len * 3, emb_dim], (r_0, s_0, a_0, r_1, s_1, a_1, ...)
        sequence = torch.stack([ret_emb, state_emb, act_emb], dim=2).reshape(
            batch_size, self.types_in_step * seq_len, self.embedding_dim
        )

        if padding_mask is not None:
            # [batch_size, seq_len * 3], stack mask identically to fit the sequence
            padding_mask = torch.stack([padding_mask for _ in range(self.types_in_step)], dim=2).reshape(
                batch_size, self.types_in_step * seq_len
            )
        return sequence, padding_mask


class OriginalActionHead(nn.Module):
    def __init__(self, action_dim: int, embedding_dim: int = 128, max_action: float = 1.0, *args, **kwargs) -> None:
        super().__init__()
        self.action_head = nn.Sequential(nn.Linear(embedding_dim, action_dim), nn.Tanh())
        self.max_action = max_action

    def forward(self, x: torch.Tensor, *args, **kwargs):
        # [batch_size, seq_len, action_dim]
        # predict actions only from state embeddings
        x = x[:, 1::3]
        return self.action_head(x) * self.max_action


class OriginalEmbeddingOutput(BaseInputOutput):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.forward_embed = OriginalEnvEmbedding(*args, **kwargs)
        self.forward_output = OriginalActionHead(*args, **kwargs)
