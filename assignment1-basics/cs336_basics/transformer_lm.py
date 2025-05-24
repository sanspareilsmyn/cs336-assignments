import torch
import torch.nn as nn

from jaxtyping import Float, Int

from torch import Tensor, einsum
from torch.nn import init


class Linear(nn.Module):
    def __init__(self, d_in: int, d_out: int, device=None, dtype=None) -> None:
        super().__init__()

        self.d_in = d_in
        self.d_out = d_out

        factory_kwargs = {'device': device, 'dtype': dtype}
        self.W = nn.Parameter(torch.empty((d_out, d_in), **factory_kwargs))

        init.trunc_normal_(self.W, mean=0.0, std=1.0)

    def forward(self, x: Tensor) -> Tensor:
        return einsum("...i, oi -> ...o", x, self.W)


def run_linear_module(
        d_in: int,
        d_out: int,
        weights: Float[Tensor, " d_out d_in"],
        in_features: Float[Tensor, " ... d_in"],
) -> Float[Tensor, " ... d_out"]:
    factory_kwargs = {'device': weights.device, 'dtype': weights.dtype}
    linear_layer = Linear(d_in=d_in, d_out=d_out, **factory_kwargs)

    state_dict_to_load = {'W': weights}
    linear_layer.load_state_dict(state_dict_to_load, strict=False)

    return linear_layer(in_features)


class Embedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, device=None, dtype=None) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model

        factory_kwargs = {'device': device, 'dtype': dtype}
        self.W = nn.Parameter(torch.empty((vocab_size, d_model), **factory_kwargs))

        init.trunc_normal_(self.W, mean=0.0, std=1.0)

    def forward(self, token_ids: Int[Tensor, " ..."]) -> Float[Tensor, " ... d_model"]:
        return self.W[token_ids]


def run_embedding_module(vocab_size: int,
                         d_model: int,
                         weights: Float[Tensor, " vocab_size d_model"],
                         token_ids: Int[Tensor, " ..."],
                         ) -> Float[Tensor, " ... d_model"]:
    factory_kwargs = {'device': weights.device, 'dtype': weights.dtype}
    embedding_layer = Embedding(vocab_size=vocab_size, d_model=d_model, **factory_kwargs)

    state_dict_to_load = {'W': weights}
    embedding_layer.load_state_dict(state_dict_to_load, strict=False)
    return embedding_layer(token_ids)


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6, device=None, dtype=None) -> None:
        super().__init__()
        self.d_model = d_model
        self.eps = eps

        factory_kwargs = {'device': device, 'dtype': dtype}
        self.W = nn.Parameter(torch.empty((d_model,), **factory_kwargs))

        init.trunc_normal_(self.W, mean=1.0, std=0.02)

    def forward(self, x: Float[Tensor, " ... d_model"]) -> Float[Tensor, " ... d_model"]:
        norm = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x * (self.W / norm)


def run_rmsnorm_module(
        d_model: int,
        eps: float,
        weights: Float[Tensor, " d_model"],
        in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    factory_kwargs = {'device': weights.device, 'dtype': weights.dtype}
    rmsnorm_layer = RMSNorm(d_model=d_model, eps=eps, **factory_kwargs)

    state_dict_to_load = {'W': weights}
    rmsnorm_layer.load_state_dict(state_dict_to_load, strict=False)

    return rmsnorm_layer(in_features)
