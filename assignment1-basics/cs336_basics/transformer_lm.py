import torch
import torch.nn as nn
from einops import rearrange

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


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff_override: int = None, device=None, dtype=None) -> None:
        super().__init__()
        self.d_model = d_model
        factory_kwargs = {'device': device, 'dtype': dtype}

        if d_ff_override is not None:
            self.d_ff = d_ff_override
        else:
            hidden_dim_approx = int((8 / 3) * d_model)
            self.d_ff = round(hidden_dim_approx / 64) * 64
            if self.d_ff == 0:
                self.d_ff = 64

        self.w1 = nn.Linear(d_model, self.d_ff, bias=False, **factory_kwargs)  # W1x
        self.w3 = nn.Linear(d_model, self.d_ff, bias=False, **factory_kwargs)  # W3x (for gating)
        self.w2 = nn.Linear(self.d_ff, d_model, bias=False, **factory_kwargs)  # Output projection

    def _silu(self, x: Tensor) -> Tensor:
        return x * torch.sigmoid(x)

    def forward(self, x: Float[Tensor, " ... d_model"]) -> Float[Tensor, " ... d_model"]:
        # 1. W1x and W3x
        hidden_states_w1 = self.w1(x)  # (..., d_ff)
        hidden_states_w3 = self.w3(x)  # (..., d_ff)

        # 2. SiLU(W1x)
        activated_states = self._silu(hidden_states_w1)

        # 3. SiLU(W1x) ⊙ W3x (Gated Linear Unit part)
        gated_states = activated_states * hidden_states_w3

        # 4. W2(gated_states)
        output = self.w2(gated_states)  # (..., d_model)

        return output


def run_swiglu_module(
        d_model: int,
        d_ff: int,
        w1_weight: Float[Tensor, " d_ff d_model"],
        w2_weight: Float[Tensor, " d_model d_ff"],
        w3_weight: Float[Tensor, " d_ff d_model"],
        in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    factory_kwargs = {'device': w1_weight.device, 'dtype': w1_weight.dtype}
    ffn_layer = PositionwiseFeedForward(d_model=d_model, d_ff_override=d_ff, **factory_kwargs)

    state_dict_to_load = {
        'w1.weight': w1_weight,
        'w2.weight': w2_weight,
        'w3.weight': w3_weight,
    }
    ffn_layer.load_state_dict(state_dict_to_load)

    return ffn_layer(in_features)


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, d_k: int, theta: float, max_seq_len: int) -> None:
        super().__init__()
        if d_k % 2 != 0:
            raise ValueError(f"d_k (dimension) must be even, but got {d_k}")

        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        dim_indices = torch.arange(0, d_k, 2, dtype=torch.float32)
        inv_freq = 1.0 / (theta ** (dim_indices / d_k))

        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)

        self.register_buffer('cos_cached', freqs.cos(), persistent=False)
        self.register_buffer('sin_cached', freqs.sin(), persistent=False)

    def forward(self, x: Float[Tensor, "... seq_len d_k"], token_positions: Int[Tensor, " ... seq_len"]) -> Float[
        Tensor, " ... seq_len d_k"]:
        token_positions_long = token_positions.long()
        cos_selected = self.cos_cached[token_positions_long]  # (..., s, d_half)
        sin_selected = self.sin_cached[token_positions_long]  # (..., s, d_half)

        # x -> (..., s, d_half, 2)
        x_pair = rearrange(x, '... s (d_half two) -> ... s d_half two', two=2)

        # Rotate
        x_even = x_pair[..., 0]
        x_odd = x_pair[..., 1]

        x_rotated_even = x_even * cos_selected - x_odd * sin_selected
        x_rotated_odd = x_even * sin_selected + x_odd * cos_selected

        result_pair = torch.empty_like(x_pair)
        result_pair[..., 0] = x_rotated_even
        result_pair[..., 1] = x_rotated_odd

        return rearrange(result_pair, '... s d_half two -> ... s (d_half two)')


def run_rope_module(
        d_k: int,
        theta: float,
        max_seq_len: int,
        in_query_or_key: Float[Tensor, " ... sequence_length d_k"],
        token_positions: Int[Tensor, " ... sequence_length"],
) -> Float[Tensor, " ... sequence_length d_k"]:
    rope_layer = RotaryPositionalEmbedding(
        d_k=d_k,
        theta=theta,
        max_seq_len=max_seq_len)
    output_tensor = rope_layer(in_query_or_key, token_positions)

    return output_tensor
