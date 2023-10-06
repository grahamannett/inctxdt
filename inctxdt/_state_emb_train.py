import torch
import torch.nn as nn

from fast_pytorch_kmeans import KMeans


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
    """
    use like:
        state_iter = [torch.from_numpy(v["observations"]) for v in dataset.dataset]
        model.train_new_state_emb(
            state_iter=state_iter, new_state_dim=17, n_clusters=1024, config=config, num_iters=10, temperature=1.0
        )
    """
    self._prev_state_emb = self.embed_paths.state_emb
    _prev_weights = self._prev_state_emb.weight.detach().clone()
    new_state_emb = nn.Linear(new_state_dim, self.embedding_dim).to(device)
    self.embed_paths.state_emb = new_state_emb

    # loss_fn = loss_fn or
    def _default_loss_fn(embeds, centroids, labels=None):
        labels = labels or torch.arange(len(embeds), device=embeds.device)

        logits = embeds @ centroids.T.to(embeds.device)
        loss = F.cross_entropy(logits / temperature, labels, reduction=reduction)
        return loss

    loss_fn = loss_fn or _default_loss_fn

    kmeans = KMeans(n_clusters=n_clusters, mode=mode, verbose=1)

    optimizer = torch.optim.AdamW(
        self.embed_paths.state_emb.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=config.betas,
    )

    _ = kmeans.fit_predict(self._prev_state_emb.weight.T.detach())

    for n_iter in range(num_iters):
        loss = 0

        for state_batch in state_iter:
            embeds = self.embed_paths.state_emb(state_batch.to(device))

            # ~infonce
            optimizer.zero_grad()
            batch_loss = loss_fn(embeds, kmeans.centroids)

            batch_loss.backward()
            optimizer.step()

            loss += batch_loss.item()
        print(f"iter:{n_iter} loss {loss}")
