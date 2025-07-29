import math
import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float
from torch.nn import Linear
from einops import rearrange
from einops import einsum   
import einx

from cs336_basics.model.rope import RotaryPositionalEmbedding

def run_softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    """
    Given a tensor of inputs, return the output of softmaxing the given `dim`
    of the input.

    Args:
        in_features (Float[Tensor, "..."]): Input features to softmax. Shape is arbitrary.
        dim (int): Dimension of the `in_features` to apply softmax to.

    Returns:
        Float[Tensor, "..."]: Tensor of with the same shape as `in_features` with the output of
        softmax normalizing the specified `dim`.
    """
    dim_max = torch.amax(in_features, dim=dim, keepdim=True)
    dim_exp = torch.exp(in_features - dim_max)
    sum_dim_exp = torch.sum(dim_exp, dim=dim, keepdim=True)
    return dim_exp / sum_dim_exp
def run_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Float[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """
    给定键(K)、查询(Q)和值(V)张量，返回缩放点积注意力实现的输出。

    参数:
        Q (Float[Tensor, " ... queries d_k"]): 查询张量
        K (Float[Tensor, " ... keys d_k"]): 键张量
        V (Float[Tensor, " ... values d_v"]): 值张量
        mask (Float[Tensor, " ... queries keys"] | None): 掩码张量
    返回:
        Float[Tensor, " ... queries d_v"]: SDPA的输出
    """
    score = torch.einsum("... q d, ... k d -> ... q k", Q, K) / math.sqrt(K.size(-1))
    # score = (Q @ K.transpose(-2, -1)) * (1.0 / math.sqrt(K.size(-1)))
    if mask is not None:
        score = score.masked_fill(mask == 0.0, float('-inf'))
        # score = score.masked_fill(mask == False, float('-inf'))
    score = run_softmax(score, dim=-1)
    att = score @ V
    return att

class Multihead_self_attention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, pos_encode: RotaryPositionalEmbedding | None = None, theta: float | None = None):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = self.d_k
        self.q_proj = Linear(self.d_model, self.num_heads * self.d_k)
        self.k_proj = Linear(self.d_model, self.num_heads * self.d_k)
        self.v_proj = Linear(self.d_model, self.num_heads * self.d_v)
        self.o_proj = Linear(self.num_heads * self.d_v, self.d_model)
        self.pos_encode = pos_encode
        self.theta = theta

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        *b, sequence_length, d_model = x.size()
        assert d_model == self.d_model
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Take Q, K, V to shape (..., num_heads, seq_len, d_k)
        Q = rearrange(Q, "... seq (heads d) -> ... heads seq d", heads=self.num_heads)
        K = rearrange(K, "... seq (heads d) -> ... heads seq d", heads=self.num_heads)
        V = rearrange(V, "... seq (heads d) -> ... heads seq d", heads=self.num_heads)
        if token_positions is None:
            token_positions = einx.rearrange("seq -> b... seq", torch.arange(sequence_length, device=x.device), b=[1] * len(b))
        token_positions = rearrange(token_positions, "... seq -> ... 1 seq")
        if self.theta is not None:
            Q = self.pos_encode(Q, token_positions)
            K = self.pos_encode(K, token_positions)
        # Construct causal mask
        causal_mask = torch.tril(torch.ones(sequence_length, sequence_length, device=x.device))
        causal_mask = causal_mask.view(1, 1, sequence_length, sequence_length)

        att = run_scaled_dot_product_attention(K=K, Q=Q, V=V, mask=causal_mask)
        # (..., sequence_length, num_heads * d_v).
        att = rearrange(att, "batch heads seq d_v -> batch seq (heads d_v)").contiguous()
        out = self.o_proj(att)
        return out