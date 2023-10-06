import torch
import torch.nn as nn
from sklearn.preprocessing import KBinsDiscretizer
import numpy as np


class Tokenizer:
    def __init__(self, num_bins: int, strategy: str = "quantile", per_column: bool = False):
        super().__init__()

        self.num_bins = num_bins
        self.strategy = strategy
        self.per_column = per_column

        self.extra_size = 2
        self.start_offset = 1

    def __call__(self, data: torch.Tensor, dtype: torch.dtype = torch.long) -> torch.Tensor:
        return self._encode_from_bins(data, dtype=dtype)

    def create(self, data: np.ndarray, device: str = "cpu"):
        self.enc = KBinsDiscretizer(n_bins=self.num_bins, encode="ordinal", strategy=self.strategy)

        if not self.per_column:
            data = data.reshape(-1, 1)
        self.enc.fit(data)

        bin_edges = [torch.from_numpy(b).to(device) for b in self.enc.bin_edges_]

        # lengths is necessary for offsetting the bin edges
        # since bin_edges may not be the same shape due to quantile strategy, keep as list
        lengths = [len(b) for b in bin_edges]
        for b_i in range(1, len(lengths)):
            lengths[b_i] = lengths[b_i] + lengths[b_i - 1]

        self.lengths, self.bin_edges = lengths, bin_edges
        #
        self.token_size = self.num_bins * (len(self.bin_edges) * self.extra_size)

    def _uniform_discretizer(self, data: torch.Tensor, n_bins: int = 1024, eps: float = 1e-6):
        raise NotImplementedError
        # self.hist = torch.histogram(data.view(-1), bins=n_bins, range=(range[0] - eps, range[1] + eps))

    def _encode_from_bins(self, x: torch.Tensor, dtype: torch.dtype = torch.long):
        xt = torch.zeros_like(x, dtype=dtype, device=x.device)

        token_offset = 0
        feature_bin = self.bin_edges[0]
        for jj in range(x.shape[-1]):
            if self.per_column:
                feature_bin = self.bin_edges[jj]
                token_offset = self.lengths[jj]

            xt[..., jj] = torch.searchsorted(feature_bin, x[..., jj].contiguous(), side="right") + token_offset

        # offset so that 0 is reserved for padding
        xt += self.start_offset
        return xt


class ModalTokenizers:
    tokenizers = {}

    def __init__(self, device: str = "cpu"):
        super().__init__()
        self.device = device
        # self.tokenizers = nn.ModuleDict()

    def __len__(self):
        return len(self.tokenizers)

    def __getitem__(self, modal: str):
        return self.tokenizers[modal]

    def new_tokenizer(
        self, modal: str, data: np.ndarray, num_bins: int, strategy: str = "quantile", per_column: bool = False
    ):
        tokenizer = Tokenizer(num_bins=num_bins, strategy=strategy, per_column=per_column)
        tokenizer.create(data, device=self.device)
        self.tokenizers[modal] = tokenizer


class OldModalTokenizers:
    # --- from model
    def set_discretizer(self, type_name: str, encoder=None, bin_edges=None, device: str = None):
        if not hasattr(self, "discretizers"):
            self.discretizers = {}

        if encoder:
            group = ["encoder", encoder]
        if bin_edges:
            bin_edges = [b.to(device) for b in bin_edges]
            lengths = [len(b) for b in bin_edges]

            for b_i in range(1, len(lengths)):
                lengths[b_i] = lengths[b_i] + lengths[b_i - 1]

            group = ["bin_edges", bin_edges, lengths]

        self.discretizers[type_name] = group

    def _encode_from_enc(self, type_name: str, x: torch.Tensor, dtype: torch.dtype = torch.long):
        _, enc = self.discretizers[type_name]
        bs, seq_len, dim = x.shape
        output = torch.tensor(enc.transform(x.reshape(-1, dim).cpu().detach()), dtype=dtype, device=x.device).reshape(
            bs, seq_len, dim
        )
        return output

    # _fn = {
    #     "encoder": _encode_from_enc,
    #     "bin_edges": _encode_from_bins,
    # }

    def encode(self, type_name: str, x: torch.Tensor, dtype: torch.dtype = torch.long):
        return self._fn[self.discretizers[type_name][0]](self, type_name, x, dtype=dtype)

    def decode(self, x: torch.Tensor, type_name: str):
        return self.discretizers[type_name].bin_edges.to(x.device)[x]
