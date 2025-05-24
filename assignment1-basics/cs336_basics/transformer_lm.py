import torch
import torch.nn as nn

from jaxtyping import Float

from torch import Tensor, einsum


class Linear(nn.Module):
    def __init__(self, d_in: int, d_out: int, device=None, dtype=None) -> None:
        super().__init__()

        self.d_in = d_in
        self.d_out = d_out

        factory_kwargs = {'device': device, 'dtype': dtype}
        self.W = nn.Parameter(torch.empty((d_out, d_in), **factory_kwargs))

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


class LinearBulitin(nn.Module):
    def __init__(self, d_in: int, d_out: int, device=None, dtype=None) -> None:
        super().__init__()
        self.linear_layer = nn.Linear(d_in, d_out, bias=False, device=device, dtype=dtype)

    def forward(self, x: Float[Tensor, " ... d_in"]) -> Float[Tensor, " ... d_out"]:
        return self.linear_layer(x)


def run_linear_module_builtin(
        d_in: int,
        d_out: int,
        weights: Float[Tensor, " d_out d_in"],
        in_features: Float[Tensor, " ... d_in"],
) -> Float[Tensor, " ... d_out"]:
    factory_kwargs = {'device': weights.device, 'dtype': weights.dtype}
    linear_layer_builtin = LinearBulitin(d_in=d_in, d_out=d_out, **factory_kwargs)

    state_dict_to_load = {'linear_layer.weight': weights}
    linear_layer_builtin.load_state_dict(state_dict_to_load, strict=False)

    return linear_layer_builtin(in_features)
