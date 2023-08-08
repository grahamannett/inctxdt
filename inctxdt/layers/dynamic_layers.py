import torch
import torch.nn as nn
from fast_pytorch_kmeans import KMeans

from inctxdt.config import EnvSpec
from inctxdt.layers.base_layers import OriginalEnvEmbedding, OriginalActionHead, BaseInputOutput


class DynamicEmbedding(nn.Module):
    def __init__(
        self,
        env_spec: EnvSpec = None,
        episode_len: int = None,
        seq_len: int = None,
        state_dim: int = None,
        action_dim: int = None,
        embedding_dim: int = 128,
        *args,
        **kwargs,
    ):
        super().__init__()
        embedding_dim = embedding_dim or getattr(env_spec, "embedding_dim", None)
        episode_len = episode_len or getattr(env_spec, "episode_len", None)
        seq_len = seq_len or getattr(env_spec, "seq_len", None)
        action_dim = action_dim or getattr(env_spec, "action_dim", None)
        state_dim = state_dim or getattr(env_spec, "state_dim", None)

        assert all((embedding_dim, episode_len, seq_len, action_dim, state_dim)), "must provide all dims or env_spec"
        self.embedding_dim, self.episode_len, self.seq_len = embedding_dim, episode_len, seq_len

        self.env_specs = {env_spec.env_name: env_spec}
        self.env_embeddings = nn.ModuleDict(
            {
                env_spec.env_name: OriginalEnvEmbedding(episode_len, seq_len, state_dim, action_dim, embedding_dim),
            }
        )

        self.base_env_name = env_spec.env_name
        # self.base_emb = self.env_embeddings[self.base_env_name]

    @torch.no_grad()
    def make_env_clusters(self, n_clusters: int = 100, mode: str = "cosine"):
        """setup base is used when we need to get the clusters

        makes the clusters

        mode must be one of ["cosine", "euclidean"]
        """
        assert mode in ["cosine", "euclidean"], "mode must be one of ['cosine', 'euclidean']"

        state_info = ("state", self.env_embeddings[self.base_env_name].state_emb.weight)
        action_weights = ("action", self.env_embeddings[self.base_env_name].action_emb.weight)
        return_weights = ("return", self.env_embeddings[self.base_env_name].return_emb.weight)
        timestep_weights = ("timestep", self.env_embeddings[self.base_env_name].timestep_emb.weight)

        cluster_info = {}
        for weight_name, weight in [state_info, action_weights, return_weights, timestep_weights]:
            kmeans = KMeans(n_clusters=n_clusters, mode=mode, verbose=1)
            _ = kmeans.fit_predict(weight.T)
            cluster_info[weight_name] = kmeans

        return cluster_info

    def new_embedding_layer(self, env_name: str, **kwargs):
        env_spec = EnvSpec(episode_len=self.episode_len, seq_len=self.seq_len, env_name=env_name)
        new_embedding_layer = OriginalEnvEmbedding(
            episode_len=self.episode_len,
            seq_len=self.seq_len,
            action_dim=env_spec.action_dim,
            state_dim=env_spec.state_dim,
            embedding_dim=self.embedding_dim,
        )

        new_embedding_layer = new_embedding_layer.to(next(self.enb_embeddings[self.base_env_name].parameters()).device)
        self.env_embeddings[env_name] = new_embedding_layer
        # new_embedding_layer.unsupervised_train(self.env_specs[self.base_env])

    def forward(self, env_name: str = None, *args, **kwargs):
        env_name = env_name or self.base_env_name

        if env_name not in self.env_embeddings:
            self.new_embedding_layer(env_name, **kwargs)

        return self.env_embeddings[env_name](*args, **kwargs)


class DynamicOutput(nn.Module):
    def __init__(
        self,
        env_spec: EnvSpec = None,
        episode_len: int = None,
        seq_len: int = None,
        action_dim: int = None,
        embedding_dim: int = 128,
    ):
        super().__init__()
        embedding_dim = embedding_dim or getattr(env_spec, "embedding_dim", None)
        seq_len = seq_len or getattr(env_spec, "seq_len", None)
        action_dim = action_dim or getattr(env_spec, "action_dim", None)
        episode_len = episode_len or getattr(env_spec, "episode_len", None)

        self.embedding_dim, self.episode_len, self.seq_len = embedding_dim, episode_len, seq_len

        self.env_specs = {env_spec.env_name: env_spec}
        self.env_output = nn.ModuleDict(
            {
                env_spec.env_name: OriginalActionHead(action_dim=action_dim, embedding_dim=embedding_dim),
            }
        )

        self.base_env_name = env_spec.env_name
        self.base_emb = self.env_output[self.base_env_name]

    @torch.no_grad()
    def make_env_clusters(self, n_clusters: int = 100, mode: str = "cosine"):
        assert mode in ["cosine", "euclidean"], "mode must be one of ['cosine', 'euclidean']"
        action_info = ("action_output", self.env_output[self.base_env_name].action_output.weight)
        cluster_info = {}
        for weight_name, weight in [action_info]:
            kmeans = KMeans(n_clusters=n_clusters, mode=mode, verbose=1)
            _ = kmeans.fit_predict(weight.T)
            cluster_info[weight_name] = kmeans

        return cluster_info

    def new_output_layer(self, env_name: str, **kwargs):
        env_spec = EnvSpec(episode_len=self.episode_len, seq_len=self.seq_len, env_name=env_name)
        new_output_layer = OriginalActionHead(action_dim=env_spec.action_dim, embedding_dim=self.embedding_dim)

        new_output_layer = new_output_layer.to(next(self.env_output[self.base_env_name].parameters()).device)
        self.env_output[env_name] = new_output_layer

    def forward(self, env_name: str = None, *args, **kwargs):
        env_name = env_name or self.base_env_name
        if env_name not in self.env_output:
            self.new_output_layer(env_name, **kwargs)

        return self.env_output[env_name](*args, **kwargs)


class DynamicLayers(BaseInputOutput):
    def __init__(
        self,
        env_spec: EnvSpec = None,
        embedding_dim: int = 128,
        **kwargs,
    ):
        super().__init__()
        self.forward_embed = DynamicEmbedding(env_spec=env_spec, embedding_dim=embedding_dim)
        self.forward_output = DynamicOutput(env_spec=env_spec, embedding_dim=embedding_dim)

    def forward(self, env_name: str, *args, **kwargs):
        breakpoint()
        sequence, padding_mask = self.forward_embed(env_name, *args, **kwargs)
        return self.forward_output(env_name, sequence)
