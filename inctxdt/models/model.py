from typing import Optional

import torch
import torch.nn as nn

from inctxdt.config import EnvSpec, ModalEmbedConfig
from inctxdt.models import layers
from inctxdt.models.model_output import ModelOutput


# Decision Transformer implementation
class TransformerBlock(nn.Module):
    def __init__(
        self,
        seq_len: int,
        embedding_dim: int,
        num_heads: int,
        attention_dropout: float,
        residual_dropout: float,
        feedforward_scale: int = 4,
    ):
        super().__init__()
        self.feedforward_dim = feedforward_scale * embedding_dim
        self.seq_len = seq_len
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.drop = nn.Dropout(residual_dropout)

        self.attention = nn.MultiheadAttention(embedding_dim, num_heads, attention_dropout, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, self.feedforward_dim),
            nn.GELU(),
            nn.Linear(self.feedforward_dim, embedding_dim),
            nn.Dropout(residual_dropout),
        )
        # True value indicates that the corresponding position is not allowed to attend
        self.register_buffer("causal_mask", ~torch.tril(torch.ones(seq_len, seq_len)).to(torch.bool))

    # [batch_size, seq_len, emb_dim] -> [batch_size, seq_len, emb_dim]
    def get_casual_mask(self, x: torch.Tensor) -> torch.Tensor:
        """casaul mask but allow for longer than seq_len"""
        if x.shape[1] > self.causal_mask.shape[0]:
            return ~torch.tril(torch.ones(x.shape[1], x.shape[1], dtype=bool)).to(x.device)
        return self.causal_mask[: x.shape[1], : x.shape[1]]

    def forward(self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        causal_mask = self.causal_mask[: x.shape[1], : x.shape[1]]

        norm_x = self.norm1(x)
        attention_out = self.attention(
            query=norm_x,
            key=norm_x,
            value=norm_x,
            attn_mask=causal_mask,
            key_padding_mask=padding_mask,
            need_weights=False,
        )[0]

        # by default pytorch attention does not use dropout
        # after final attention weights projection, while minGPT does:
        # https://github.com/karpathy/minGPT/blob/7218bcfa527c65f164de791099de715b81a95106/mingpt/model.py#L70 # noqa
        x = x + self.drop(attention_out)
        x = x + self.mlp(self.norm2(x))
        return x


class DecisionTransformer(nn.Module):
    def __init__(
        self,
        state_dim: int = None,
        action_dim: int = None,
        seq_len: int = 200,
        episode_len: int = 4096,
        embedding_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        attention_dropout: float = 0.0,
        residual_dropout: float = 0.0,
        embedding_dropout: float = 0.0,
        max_action: float = 1.0,
        env_spec: Optional["EnvSpec"] = None,
        modal_embed: ModalEmbedConfig = None,
        discretizers: dict = {},
        **kwargs,
    ):
        super().__init__()

        # try to get from env_spec as that is the most reliable/likely
        state_dim = getattr(env_spec, "state_dim", state_dim)
        action_dim = getattr(env_spec, "action_dim", action_dim)
        # base params
        self.seq_len = seq_len
        self.embedding_dim = embedding_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.episode_len = episode_len
        self.max_action = max_action

        self.discretizers = discretizers

        # embedding drop/norm
        self.emb_drop = nn.Dropout(embedding_dropout)
        self.emb_norm = nn.LayerNorm(embedding_dim)
        self.out_norm = nn.LayerNorm(embedding_dim)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    seq_len=episode_len,
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    attention_dropout=attention_dropout,
                    residual_dropout=residual_dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self.modal_embed_config = modal_embed or ModalEmbedConfig(embedding_dim=embedding_dim)
        #  Layers are things like - SequentialAction  - StackedEnvEmbedding
        EmbedClass = getattr(layers, self.modal_embed_config.EmbedClass)
        self.embed_paths = EmbedClass(
            modal_embed_config=self.modal_embed_config,
            embedding_dim=embedding_dim,
            episode_len=episode_len,
            seq_len=seq_len,
            state_dim=state_dim,
            action_dim=action_dim,
            max_action=max_action,
        )
        self.embed_paths.patch_parent(parent=self)

        self.observation_head = nn.Sequential(nn.Linear(self.embedding_dim, state_dim), nn.Tanh())
        self.reward_head = nn.Sequential(nn.Linear(self.embedding_dim, 1))

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(
        self,
        states: torch.Tensor,  # [batch_size, seq_len, state_dim]
        actions: torch.Tensor,  # [batch_size, seq_len, action_dim]
        returns_to_go: torch.Tensor,  # [batch_size, seq_len]
        timesteps: torch.Tensor,  # [batch_size, seq_len]
        mask: Optional[torch.Tensor] = None,  # [batch_size, seq_len]
        padding_mask: Optional[torch.Tensor] = None,  # [batch_size, seq_len]
        **kwargs,
    ) -> torch.FloatTensor:
        bs, seq_len = states.shape[0], states.shape[1]

        if self.modal_embed_config.tokenize_action:
            actions = self.discretizers.actions(actions)

        # sequence should be:
        #   [batch_size, seq_len * spread_dim, embedding_dim]
        sequence = self.forward_embed(
            states=states,
            actions=actions,
            returns_to_go=returns_to_go,
            timesteps=timesteps,
            **kwargs,
        )

        spread_dim = sequence.shape[1] // seq_len
        if padding_mask is not None:
            # padding mask comes in as [bs, seq_len] -> want [bs, unpacked_sequence_len]
            padding_mask = torch.stack([padding_mask for _ in range(spread_dim)], dim=-1).reshape(bs, -1)

        out = self.emb_norm(sequence)
        out = self.emb_drop(out)

        for block in self.blocks:
            out = block(out, padding_mask=padding_mask)

        # norm and reshape to [batch_size, seq_len, spread_dim, embedding_dim]
        out = self.out_norm(out)
        out = out.reshape(
            bs, seq_len, spread_dim, self.embedding_dim
        )  # [batch_size, seq_len, spread_dim, embedding_dim]

        act_logits = self.forward_output(out) * self.max_action

        # extra
        obs_logits = self.observation_head(out[:, :, 0, :])
        rewards_out = self.reward_head(out[:, :, -1, :]).squeeze(-1)

        return ModelOutput(logits=act_logits, extra={"obs_logits": obs_logits, "rewards": rewards_out})


if __name__ == "__main__":
    batch_size, seq_len, state_dim, action_dim = 4, 10, 17, 6
    states = torch.rand(batch_size, seq_len, state_dim)
    actions = torch.rand(batch_size, seq_len, action_dim)
    returns_to_go = torch.rand(batch_size, seq_len)
    timesteps = torch.arange(seq_len, dtype=torch.long).view(1, -1).repeat(batch_size, 1)

    padding_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

    model = DecisionTransformer(
        state_dim=state_dim,
        action_dim=action_dim,
        seq_len=seq_len,
        episode_len=1000,
        embedding_dim=128,
        num_layers=2,
        num_heads=2,
    )
    out = model(states=states, actions=actions, returns_to_go=returns_to_go, timesteps=timesteps)
