import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, einsum
from jaxtyping import Float, Int
from torch import Tensor
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


class SoftmaxModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor, dim: int) -> Tensor:
        x_max = torch.max(x, dim=dim, keepdim=True).values
        x_stabilized = x - x_max
        exp_x = torch.exp(x_stabilized)
        sum_exp_x = torch.sum(exp_x, dim=dim, keepdim=True)
        return exp_x / sum_exp_x


def run_softmax_module(
        input_tensor: Tensor,
        dim: int,
) -> Tensor:
    softmax_layer = SoftmaxModule()
    return softmax_layer(input_tensor, dim=dim)


def run_scaled_dot_product_attention_module(
        Q: Tensor,  # "... q d_k"
        K: Tensor,  # "... k d_k"
        V: Tensor,  # "... k d_v"
        mask: Tensor | None = None,
) -> Tensor:  # "... q d_v"
    d_k = Q.shape[-1]

    # 1. Inner product of Q and K
    scores = einsum(Q, K, '... q_len d_k, ... k_len d_k -> ... q_len k_len')

    # 2. Scaling by sqrt(d_k)
    scores = scores / math.sqrt(d_k)

    # 3. Masking
    if mask is not None:
        if mask.dtype == torch.bool:
            condition_to_fill = (mask == False)
        else:
            condition_to_fill = (mask == 0.0)
        scores = scores.masked_fill(condition_to_fill, float('-inf'))

    attention_weights = F.softmax(scores, dim=-1)

    # 5. Inner product of attention weights and V
    output = einsum(attention_weights, V, '... q_len k_len, ... k_len d_v -> ... q_len d_v')

    return output


def run_multihead_self_attention_module(
        d_model: int,
        num_heads: int,
        q_proj_weight: Tensor,  # (d_model, d_model)
        k_proj_weight: Tensor,  # (d_model, d_model)
        v_proj_weight: Tensor,  # (d_model, d_model)
        o_proj_weight: Tensor,  # (d_model, d_model)
        in_features: Tensor,  # (..., sequence_length, d_model)
) -> Tensor:  # (..., sequence_length, d_model)
    # Dimension per head
    d_k_per_head = d_model // num_heads
    d_v_per_head = d_model // num_heads

    # Sequence length
    seq_len = in_features.shape[-2]

    # 1. Linear projections
    q_raw = F.linear(in_features, q_proj_weight)
    k_raw = F.linear(in_features, k_proj_weight)
    v_raw = F.linear(in_features, v_proj_weight)

    # 2. Reshape Q_raw, K_raw, V_raw for multi-head
    # (..., seq_len, d_model) -> (..., num_heads, seq_len, d_k_per_head)
    Q_multi_head = rearrange(q_raw, '... s (h d) -> ... h s d', h=num_heads, d=d_k_per_head)
    K_multi_head = rearrange(k_raw, '... s (h d) -> ... h s d', h=num_heads, d=d_k_per_head)
    V_multi_head = rearrange(v_raw, '... s (h d) -> ... h s d', h=num_heads, d=d_v_per_head)

    # 3. Create Causal Mask
    causal_mask = torch.ones(seq_len, seq_len, device=in_features.device, dtype=torch.bool).tril(diagonal=0)

    # 4. Apply Scaled Dot-Product Attention using the dedicated function
    attn_output_per_head = run_scaled_dot_product_attention_module(
        Q_multi_head, K_multi_head, V_multi_head, mask=causal_mask
    )

    # 5. Concatenate heads
    attn_output_concatenated = rearrange(attn_output_per_head, '... h s d -> ... s (h d)')

    # 6. Final linear projection
    output = F.linear(attn_output_concatenated, o_proj_weight)

    return output


def run_multihead_self_attention_with_rope_module(
        d_model: int,
        num_heads: int,
        max_seq_len: int,
        theta: float,
        q_proj_weight: Float[Tensor, " d_k d_in"],
        k_proj_weight: Float[Tensor, " d_k d_in"],
        v_proj_weight: Float[Tensor, " d_v d_in"],
        o_proj_weight: Float[Tensor, " d_model d_v"],
        in_features: Float[Tensor, " ... sequence_length d_in"],
        token_positions: Int[Tensor, " ... sequence_length"] | None = None,
) -> Float[Tensor, " ... sequence_length d_out"]:
    # Dimension per head
    d_k_per_head = d_model // num_heads
    d_v_per_head = d_model // num_heads

    # Sequence length
    leading_dims = in_features.shape[:-2]
    seq_len = in_features.shape[-2]

    # 1. Linear projections
    q_raw = F.linear(in_features, q_proj_weight)  # (..., seq_len, d_model)
    k_raw = F.linear(in_features, k_proj_weight)  # (..., seq_len, d_model)
    v_raw = F.linear(in_features, v_proj_weight)  # (..., seq_len, d_model)

    # 2. Reshape Q_raw, K_raw, V_raw for multi-head
    Q_multi_head_no_rope = rearrange(q_raw, '... s (h d) -> ... h s d', h=num_heads, d=d_k_per_head)
    K_multi_head_no_rope = rearrange(k_raw, '... s (h d) -> ... h s d', h=num_heads, d=d_k_per_head)
    V_multi_head = rearrange(v_raw, '... s (h d) -> ... h s d', h=num_heads, d=d_v_per_head)

    # 3. Apply RoPE to Q_multi_head and K_multi_head
    if token_positions.ndim == len(leading_dims) + 1:  # (e.g., batch_size, seq_len)
        token_pos_for_rope = token_positions.unsqueeze(-2)
    elif token_positions.ndim == len(leading_dims) + 2 and token_positions.shape[-2] == num_heads:
        token_pos_for_rope = token_positions
    elif token_positions.ndim == len(leading_dims) + 2 and token_positions.shape[-2] == 1:
        token_pos_for_rope = token_positions

    Q_rope_applied = run_rope_module(
        d_k=d_k_per_head,
        theta=theta,
        max_seq_len=max_seq_len,
        in_query_or_key=Q_multi_head_no_rope,
        token_positions=token_pos_for_rope
    )
    K_rope_applied = run_rope_module(
        d_k=d_k_per_head,
        theta=theta,
        max_seq_len=max_seq_len,
        in_query_or_key=K_multi_head_no_rope,
        token_positions=token_pos_for_rope
    )

    # 4. Create Causal Mask
    causal_mask = torch.ones(seq_len, seq_len, device=in_features.device, dtype=torch.bool).tril(diagonal=0)

    # 5. Apply Scaled Dot-Product Attention
    attn_output_per_head = run_scaled_dot_product_attention_module(
        Q_rope_applied, K_rope_applied, V_multi_head, mask=causal_mask
    )

    # 6. Concatenate heads
    attn_output_concatenated = rearrange(attn_output_per_head, '... h s d -> ... s (h d)')

    # 7. Final linear projection
    output = F.linear(attn_output_concatenated, o_proj_weight)

    return output


def run_transformer_block_module(
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        weights: dict[str, Tensor],
        in_features: Float[Tensor, "batch sequence_length d_model"],
) -> Float[Tensor, "batch sequence_length d_model"]:
    batch_size, seq_len, _ = in_features.shape

    # 0. Create token positions
    token_positions = torch.arange(seq_len, device=in_features.device).unsqueeze(0).expand(batch_size, -1)

    # --- First sublayer: Multi-Head Self-Attention ---
    # 1. RMSNorm (Pre-Normalization)
    norm1_weights = weights['ln1.weight']
    x_norm1 = run_rmsnorm_module(
        d_model=d_model,
        eps=1e-6,
        weights=norm1_weights,
        in_features=in_features
    )

    # 2. Causal Multi-Head Self-Attention with RoPE
    q_proj_w = weights['attn.q_proj.weight']
    k_proj_w = weights['attn.k_proj.weight']
    v_proj_w = weights['attn.v_proj.weight']
    o_proj_w = weights['attn.output_proj.weight']

    attn_output = run_multihead_self_attention_with_rope_module(
        d_model=d_model,
        num_heads=num_heads,
        max_seq_len=max_seq_len,
        theta=theta,
        q_proj_weight=q_proj_w,
        k_proj_weight=k_proj_w,
        v_proj_weight=v_proj_w,
        o_proj_weight=o_proj_w,
        in_features=x_norm1,
        token_positions=token_positions
    )

    # 3. Residual Connection
    x_after_attn = in_features + attn_output

    # --- Second sublayer: Feed-Forward Network ---
    # 4. RMSNorm (Pre-Normalization)
    norm2_weights = weights['ln2.weight']
    x_norm2 = run_rmsnorm_module(
        d_model=d_model,
        eps=1e-6,
        weights=norm2_weights,
        in_features=x_after_attn
    )

    # 5. Position-wise Feed-Forward Network (SwiGLU)
    ffn_w1_weight = weights['ffn.w1.weight']
    ffn_w2_weight = weights['ffn.w2.weight']
    ffn_w3_weight = weights['ffn.w3.weight']

    ffn_output = run_swiglu_module(
        d_model=d_model,
        d_ff=d_ff,
        w1_weight=ffn_w1_weight,
        w2_weight=ffn_w2_weight,
        w3_weight=ffn_w3_weight,
        in_features=x_norm2
    )

    # 6. Residual Connection
    output = x_after_attn + ffn_output

    return output
