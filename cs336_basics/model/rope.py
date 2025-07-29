import torch
from torch import nn

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        theta_ik = theta ** (-torch.arange(0, d_k, 2, device=device) / d_k) # [d_k / 2]
        pos = torch.arange(max_seq_len, device=device) # [max_seq_len]
        angles = torch.einsum("i,j->ij", pos, theta_ik) # [max_seq_len, d_k / 2]
        self.register_buffer("cos", torch.cos(angles), persistent=False)
        self.register_buffer("sin", torch.sin(angles), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        cos = self.cos[token_positions]
        sin = self.sin[token_positions]
        x1 = x[..., 0::2] # [..., d_k / 2]
        x2 = x[..., 1::2]
        rotated_x1 = x1 * cos - x2 * sin
        rotated_x2 = x1 * sin + x2 * cos
        # out = torch.stack([rotated_x1, rotated_x2], dim=-1).reshape(x.shape) # [..., d_k / 2, 2] -> [..., d_k]
        out = torch.stack([rotated_x1, rotated_x2], dim=-1).flatten(-2)
        return out