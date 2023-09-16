import torch
import torch.nn as nn


class ConvActionDimHead(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        action_dim: int,
        stack_idxs: list[int] = [0, 1],
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
