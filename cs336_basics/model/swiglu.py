import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float
from cs336_basics.model.linear import Linear

  
class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        # 移除不必要的d_ff调整，直接使用传入的d_ff
        self.d_ff = d_ff
        self.w1 = Linear(d_model, self.d_ff)
        self.w2 = Linear(self.d_ff, d_model)
        self.w3 = Linear(d_model, self.d_ff)

    def load_weights(self, w1: torch.Tensor, w2: torch.Tensor, w3: torch.Tensor):
        """加载权重，确保形状匹配"""
        with torch.no_grad():
            # 直接复制权重，不需要转置（因为Linear类已经处理了）
            self.w1.weight.data = w1
            self.w2.weight.data = w2
            self.w3.weight.data = w3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        SwiGLU前向传播：
        SwiGLU(x) = (SiLU(xW1) ⊙ xW3)W2
        """
        w1_out = self.w1(x)        # x @ W1
        w3_out = self.w3(x)        # x @ W3
        silu_out = w1_out * torch.sigmoid(w1_out)  # SiLU(xW1) 
        gated = silu_out * w3_out   # SiLU(xW1) ⊙ xW3
        return self.w2(gated)       # (SiLU(xW1) ⊙ xW3)W2
    


def run_silu(in_features: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
    """Given a tensor of inputs, return the output of applying SiLU
    to each element.

    Args:
        in_features(Float[Tensor, "..."]): Input features to run SiLU on. Shape is arbitrary.

    Returns:
        Float[Tensor,"..."]: of with the same shape as `in_features` with the output of applying
        SiLU to each element.
    """
    # 最后一维做哈达玛积
    return in_features * torch.sigmoid(in_features)



def adjust_weight(weight: torch.Tensor, target_shape: tuple) -> torch.Tensor:
    current_shape = weight.shape
    # Use the same device as weight, but default to CPU if .device is not accessible
    device = weight.device if weight.device.type != 'meta' else torch.device('cpu')
    new_weight = torch.zeros(target_shape, dtype=weight.dtype, device=device)
    min_dim0 = min(current_shape[0], target_shape[0])
    min_dim1 = min(current_shape[1], target_shape[1])
    new_weight[:min_dim0, :min_dim1] = weight[:min_dim0, :min_dim1]
    return new_weight