import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float, Int

## 理解要求
# 继承自 torch.nn.Module
# 不包含bias参数
# 权重矩阵存储为W（而不是W的转置）
# 使用 torch.nn.init.trunc_normal_ 初始化
# 不能使用PyTorch内置的 nn.Linear

class Linear(nn.Module):
        """
        线形层，无偏置
        作用是将张量的两个维度做线性变换 二维
        """
        def __init__(self,in_features,out_features,device=None,dtype=None):
            """
        构造线性变换模块
        
        参数:
            in_features: 输入特征的最后一个维度
            out_features: 输出特征的最后一个维度  
            device: 存储参数的设备
            dtype: 参数的数据类型
        """
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features
            # 创建权重参数矩阵 W: (out_features, in_features)
            # 注意：存储为W而不是W的转置，便于内存访问
            self.weight = nn.Parameter(
                torch.empty(
                    out_features,
                    in_features,
                    device=device,
                    dtype=dtype
                )
            )
            # 初始化权重
            self._reset_parameters()

        def _reset_parameters(self): 
            """
            初始化空的矩阵
            """
            # 使用截断正态分布初始化权重
            # std = sqrt(2 / (in_features + out_features)) 是常用的Xavier初始化变体
            std = (2.0 / (self.in_features + self.out_features)) ** 0.5
            torch.nn.init.trunc_normal_(self.weight, std=std)

        def forward(self,x: Tensor)-> Tensor:
            return x @ self.weight.T

def run_linear(
    d_in: int,
    d_out: int,
    weights: Float[Tensor, " d_out d_in"],
    in_features: Float[Tensor, " ... d_in"],
) -> Float[Tensor, " ... d_out"]:
    """
    给定线性层的权重，计算批量输入的变换。

    参数:
        d_in (int): 输入维度的大小
        d_out (int): 输出维度的大小
        weights (Float[Tensor, "d_out d_in"]): 要使用的线性权重
        in_features (Float[Tensor, "... d_in"]): 要应用函数的输出张量

    返回:
        Float[Tensor, "... d_out"]: 线性模块的变换输出。
    """
    
    
    # 创建Linear层实例
    linear_layer = Linear(
        in_features=d_in, 
        out_features=d_out, 
        device=weights.device, 
        dtype=weights.dtype
    )
    # 使用提供的权重替换随机初始化的权重
    with torch.no_grad():
        linear_layer.weight.data = weights
        
    return linear_layer(in_features)