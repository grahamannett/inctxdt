from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from fast_pytorch_kmeans import KMeans


from inctxdt.config import EnvSpec
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
        self.register_buffer("causal_mask", ~torch.tril(torch.ones(seq_len, seq_len)).to(bool))
        self.seq_len = seq_len

    # [batch_size, seq_len, emb_dim] -> [batch_size, seq_len, emb_dim]

    def get_casual_mask(self, x: torch.Tensor) -> torch.Tensor:
        # if x.shape[1] > self.causal_mask.shape[0]:
        #     return ~torch.tril(torch.ones(x.shape[1], x.shape[1], dtype=bool)).to(x.device)
        return self.causal_mask[: x.shape[1], : x.shape[1]]

    def forward(self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
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
        EmbedClass: Optional[str] = "SequentialAction",
        **kwargs,
    ):
        super().__init__()
        self.emb_drop = nn.Dropout(embedding_dropout)
        self.emb_norm = nn.LayerNorm(embedding_dim)

        state_dim = getattr(env_spec, "state_dim", state_dim)
        action_dim = getattr(env_spec, "action_dim", action_dim)
        # self.state_embed_dict = nn.ModuleDict({})

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

        EmbedClass = getattr(layers, EmbedClass)  # layers.SequentialAction  #  StackedEnvEmbedding
        self.embed_output_layers = EmbedClass(
            embedding_dim=embedding_dim,
            episode_len=episode_len,
            seq_len=seq_len,
            state_dim=state_dim,
            action_dim=action_dim,
        )
        self.embed_output_layers.patch_parent(parent=self)

        # self.embed_output_layers = DynamicLayers(env_spec=env_spec, embedding_dim=embedding_dim)

        self.seq_len = seq_len
        self.embedding_dim = embedding_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.episode_len = episode_len
        self.max_action = max_action

        self.apply(self._init_weights)

    def set_discretizer(self, type_name: str, bin_edges: torch.Tensor) -> None:
        if not hasattr(self, "discretizers"):
            self.discretizers = {}

        self.discretizers[type_name] = bin_edges

    def _uniform_discretizer(self, type_name: str, arr: torch.Tensor, n_bins: int = 1024, eps: float = 1e-6):
        hist = torch.histogram(arr.view(-1), bins=n_bins, range=(range[0] - eps, range[1] + eps))
        self.discretizers[type_name] = hist

    def encode(self, type_name: str, x: torch.Tensor, dtype: torch.dtype = torch.long):
        xt = torch.zeros_like(x, dtype=dtype, device=x.device)
        offset_bin_edge = 0
        for jj in range(x.shape[-1]):
            bin_edges = self.discretizers[type_name][jj]
            xt[..., jj] = torch.searchsorted(bin_edges[1:-1], x[..., jj], side="right")
            xt[..., jj] += jj * len(bin_edges)

        return xt

    def decode(self, x: torch.Tensor, type_name: str):
        return self.discretizers[type_name].bin_edges.to(x.device)[x]

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
        # if actions is not None:
        #     actions = self.encode(actions, "actions")
        actions = self.encode("actions", actions)

        sequence, padding_mask = self.forward_embed(
            states=states,
            actions=actions,
            returns_to_go=returns_to_go,
            timesteps=timesteps,
            padding_mask=padding_mask,
            **kwargs,
        )

        out = self.emb_norm(sequence)
        out = self.emb_drop(out)

        for block in self.blocks:
            out = block(out, padding_mask=padding_mask)

        logits, obs_logits = self.forward_output(x=out)

        return ModelOutput(logits=logits, extra={"obs_logits": obs_logits})

    def train_new_state_emb(
        self,
        state_iter,
        new_state_dim: int,
        n_clusters: int = 1024,
        loss_fn: callable = None,
        mode: str = "cosine",
        device: str = "cuda",
        temperature=0.5,
        reduction="mean",
        config=None,
        num_iters=10,
    ):
        self._prev_state_emb = self.embed_output_layers.state_emb
        _prev_weights = self._prev_state_emb.weight.detach().clone()
        new_state_emb = nn.Linear(new_state_dim, self.embedding_dim).to(device)
        self.embed_output_layers.state_emb = new_state_emb

        # loss_fn = loss_fn or
        def _default_loss_fn(embeds, centroids, labels=None):
            labels = labels or torch.arange(len(embeds), device=embeds.device)

            logits = embeds @ centroids.T.to(embeds.device)
            loss = F.cross_entropy(logits / temperature, labels, reduction=reduction)
            return loss

        loss_fn = loss_fn or _default_loss_fn

        kmeans = KMeans(n_clusters=n_clusters, mode=mode, verbose=1)

        optimizer = torch.optim.AdamW(
            self.embed_output_layers.state_emb.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=config.betas,
        )

        _ = kmeans.fit_predict(self._prev_state_emb.weight.T.detach())

        for n_iter in range(num_iters):
            loss = 0

            for state_batch in state_iter:
                embeds = self.embed_output_layers.state_emb(state_batch.to(device))

                # ~infonce
                optimizer.zero_grad()
                batch_loss = loss_fn(embeds, kmeans.centroids)

                batch_loss.backward()
                optimizer.step()

                loss += batch_loss.item()
            print(f"iter:{n_iter} loss {loss}")

    def generate_actions(
        self,
        timestep: int,
        states: torch.Tensor,
        actions: torch.Tensor,
        returns_to_go: torch.Tensor,  # [batch_size, seq_len]
        timesteps: torch.Tensor,  # [batch_size, seq_len]
        action_len: int = None,  # generate this many
        mask: Optional[torch.Tensor] = None,  # [batch_size, seq_len]
        padding_mask: Optional[torch.Tensor] = None,  # [batch_size, seq_len]
        **kwargs,
    ) -> torch.FloatTensor:
        raise NotImplementedError


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
