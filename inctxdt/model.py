from typing import Optional

# from gymnasium import Env
import torch
import torch.nn as nn
import torch.nn.functional as F

from inctxdt.model_output import ModelOutput


# Decision Transformer implementation
class TransformerBlock(nn.Module):
    def __init__(
        self,
        seq_len: int,
        embedding_dim: int,
        num_heads: int,
        attention_dropout: float,
        residual_dropout: float,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.drop = nn.Dropout(residual_dropout)

        self.attention = nn.MultiheadAttention(
            embedding_dim, num_heads, attention_dropout, batch_first=True
        )
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            nn.GELU(),
            nn.Linear(4 * embedding_dim, embedding_dim),
            nn.Dropout(residual_dropout),
        )
        # True value indicates that the corresponding position is not allowed to attend
        self.register_buffer(
            "causal_mask", ~torch.tril(torch.ones(seq_len, seq_len)).to(bool)
        )
        self.seq_len = seq_len

    # [batch_size, seq_len, emb_dim] -> [batch_size, seq_len, emb_dim]

    def get_casual_mask(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] > self.causal_mask.shape[0]:
            return ~torch.tril(torch.ones(x.shape[1], x.shape[1])).to(bool).to(x.device)

        return self.causal_mask[: x.shape[1], : x.shape[1]]

    def forward(
        self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # causal_mask = self.causal_mask[: x.shape[1], : x.shape[1]]
        causal_mask = self.get_casual_mask(x)

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


class EnvEmbedding(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, embedding_dim: int):
        super().__init__()
        self.state_emb = nn.Linear(state_dim, embedding_dim)
        self.action_emb = nn.Linear(action_dim, embedding_dim)
        self.return_emb = nn.Linear(1, embedding_dim)

    # def forward(self, enb_batch: Batch):
    #     pass


class DecisionTransformer(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        seq_len: int = 200,
        episode_len: int = 1000,
        embedding_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 8,
        attention_dropout: float = 0.0,
        residual_dropout: float = 0.0,
        embedding_dropout: float = 0.0,
        max_action: float = 1.0,
        use_single_action_head: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.emb_drop = nn.Dropout(embedding_dropout)
        self.emb_norm = nn.LayerNorm(embedding_dim)

        self.out_norm = nn.LayerNorm(embedding_dim)
        # additional seq_len embeddings for padding timesteps
        self.timestep_emb = nn.Embedding(episode_len + seq_len, embedding_dim)
        self.state_emb = nn.Linear(state_dim, embedding_dim)
        self.action_emb = nn.Linear(action_dim, embedding_dim)
        self.return_emb = nn.Linear(1, embedding_dim)

        self.env_embs = nn.ModuleDict()

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    seq_len=3 * seq_len,
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    attention_dropout=attention_dropout,
                    residual_dropout=residual_dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self._use_single_action_head = use_single_action_head

        if not use_single_action_head:
            self._multi_action_head_mods = nn.Sequential(
                nn.Linear(embedding_dim, action_dim), nn.Tanh()
            )

            self.action_head = self.multi_action_head

        else:
            self.single_action_head_mods = nn.ModuleDict(
                {
                    "norm": nn.LayerNorm(embedding_dim),
                    "head": nn.Linear(embedding_dim, 1),
                    "activation": nn.Tanh(),
                }
            )
            self.action_head = self.single_action_head

        self.seq_len = seq_len
        self.embedding_dim = embedding_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.episode_len = episode_len
        self.max_action = max_action

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

    def make_env_emb(self, env: "Env"):
        if env.spec.id not in self.env_embs:
            self.env_embs[env.spec.id] = EnvEmbedding(
                state_dim=env.observation_space.shape[0],
                action_dim=env.action_space.shape[0],
                embedding_dim=self.embedding_dim,
            )

    def multi_action_head(self, x: torch.Tensor, **kwargs):
        return self._multi_action_head_mods(self.out_norm(x)[:, 1::3])

    def single_action_head(
        self,
        x: torch.Tensor,
        act_dim: int = None,  # number of actions needed per pred step
        seq_len: int = None,
    ) -> torch.Tensor:
        act_dim = act_dim or self.action_dim
        seq_len = seq_len or (x.shape[1] // 3)

        x = F.adaptive_avg_pool2d(x, (act_dim * seq_len, self.embedding_dim))
        x = self.single_action_head_mods["norm"](x)
        x = self.single_action_head_mods["head"](x)
        return self.single_action_head_mods["activation"](x)

    def forward(
        self,
        states: torch.Tensor,  # [batch_size, seq_len, state_dim]
        actions: torch.Tensor,  # [batch_size, seq_len, action_dim]
        returns_to_go: torch.Tensor,  # [batch_size, seq_len]
        time_steps: torch.Tensor,  # [batch_size, seq_len]
        mask: Optional[torch.Tensor] = None,  # [batch_size, seq_len]
        padding_mask: Optional[torch.Tensor] = None,  # [batch_size, seq_len]
    ) -> torch.FloatTensor:
        batch_size, seq_len = states.shape[0], states.shape[1]
        # [batch_size, seq_len, emb_dim]
        time_emb = self.timestep_emb(time_steps)

        obs_emb = self.state_emb(states) + time_emb
        act_emb = self.action_emb(actions) + time_emb
        re_emb = self.return_emb(returns_to_go.unsqueeze(-1)) + time_emb

        # [batch_size, seq_len * 3, emb_dim], (r_0, s_0, a_0, r_1, s_1, a_1, ...)
        # sequence = (
        #     torch.stack([returns_emb, state_emb, act_emb], dim=1).permute(0, 2, 1, 3).reshape(batch_size, 3 * seq_len, self.embedding_dim)
        # )
        sequence = (
            torch.stack([obs_emb, re_emb, act_emb], dim=1)
            .permute(0, 2, 1, 3)
            .reshape(batch_size, 3 * seq_len, self.embedding_dim)
        )

        if mask is not None:
            padding_mask = ~mask.to(torch.bool)

        if padding_mask is not None:
            # [batch_size, seq_len * 3], stack mask identically to fit the sequence
            padding_mask = (
                torch.stack([padding_mask, padding_mask, padding_mask], dim=1)
                .permute(0, 2, 1)
                .reshape(batch_size, 3 * seq_len)
            )
        # LayerNorm and Dropout (!!!) as in original implementation,
        # while minGPT & huggingface uses only embedding dropout
        out = self.emb_norm(sequence)
        out = self.emb_drop(out)

        for block in self.blocks:
            out = block(out, padding_mask=padding_mask)

        # should be action dim
        # output1 = self.single_action_head(out, act_dim=actions.shape[-1])
        # output2 = self.multi_action_head(self.out_norm(out)[:, 1::3])

        output = self.action_head(out, act_dim=actions.shape[-1], seq_len=seq_len)

        return ModelOutput(logits=output)

    def predict_action(self, len_act: int, *args, **kwargs):
        for i in range(len_act):
            pass


if __name__ == "__main__":
    states = torch.rand(4, 10, 29)
    actions = torch.rand(4, 10, 8)
    returns_to_go = torch.rand(4, 10)
    time_steps = torch.arange(10, dtype=torch.long).view(1, -1).repeat(4, 1)

    model = DecisionTransformer(29, 8, num_layers=2, num_heads=2)
    out = model(states, actions, returns_to_go, time_steps)
    breakpoint()

    # torch.Size([4, 10, 29])
# (Pdb) actions.shape
# torch.Size([4, 10, 8])
# (Pdb) returns_to_go.shape
# torch.Size([4, 10])
# (Pdb) time_steps.shape
# torch.Size([4, 10])
