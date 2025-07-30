import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor

class RMSNorm(nn.Module):
    """
    RMS Layer Normalization (Root Mean Square Layer Normalization)
    
    RMSNorm 是 LayerNorm 的简化版本，只进行缩放操作，不进行平移（centering）。
    相比传统 LayerNorm，RMSNorm 计算更高效，在大模型中广泛使用。
    
    数学公式：
    RMS(x) = sqrt(mean(x^2) + eps)
    output = (x / RMS(x)) * weight
    
    其中：
    - x: 输入特征 (..., d_model)
    - weight: 可学习的缩放参数 (d_model,)
    - eps: 数值稳定性参数，防止除零
    """
    
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        """
        初始化 RMSNorm 层
        
        参数:
            d_model (int): 输入特征的最后一个维度大小
            eps (float): 数值稳定性参数，防止RMS为0时的除零错误
                        默认1e-5，通常取值范围 [1e-8, 1e-5]
            device: 参数存储的设备 (cpu/cuda)
            dtype: 参数的数据类型 (如 torch.float32)
        """
        super().__init__()
        
        # 使用 factory_kwargs 确保所有参数在相同设备和数据类型
        factory_kwargs = {'device': device, 'dtype': dtype}
        
        # 存储超参数
        self.eps = eps
        self.d_model = d_model
        
        # 可学习的缩放参数，初始化为全1
        # 形状: (d_model,) - 对应输入的最后一个维度
        self.weight = nn.Parameter(torch.ones(d_model, **factory_kwargs))

    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        """
        RMSNorm 前向传播
        
        参数:
            x: 输入张量，形状为 (..., d_model)
               可以是任意维度，但最后一维必须等于 d_model
               例如: (batch_size, seq_len, d_model) 或 (batch_size, d_model)
        
        返回:
            规范化后的张量，形状与输入相同 (..., d_model)
        
        计算流程:
        1. 保存原始数据类型
        2. 转换到 float32 以提高数值稳定性
        3. 计算 RMS (Root Mean Square)
        4. 进行规范化和缩放
        5. 转换回原始数据类型
        """
        
        # 1. 保存输入的原始数据类型，用于最后恢复
        # 这样做是为了支持混合精度训练 (如 float16)
        in_dtype = x.dtype
        
        # 2. 转换到 float32 进行计算，提高数值稳定性
        # 避免在 float16 下的数值误差和溢出问题
        x = x.to(torch.float32)
        
        # 3. 计算 RMS (Root Mean Square)
        # x.pow(2): 逐元素平方, 形状不变 (..., d_model)
        # .mean(dim=-1, keepdim=True): 沿最后一维求均值
        #   - dim=-1: 对最后一维 (d_model) 求均值
        #   - keepdim=True: 保持维度，结果形状为 (..., 1)
        # + self.eps: 加上小常数防止开根号时除零
        # torch.sqrt: 开平方根得到 RMS
        RMS_x = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        
        # 4. RMSNorm 变换
        # x / RMS_x: 归一化，将每个样本的RMS缩放到1
        #   广播机制: (..., d_model) / (..., 1) = (..., d_model)
        # * self.weight: 应用可学习的缩放参数
        #   广播机制: (..., d_model) * (d_model,) = (..., d_model)
        result = x / RMS_x * self.weight
        
        # 5. 转换回原始数据类型
        # 支持混合精度训练，保持与输入相同的精度
        return result.to(in_dtype)

def run_rmsnorm(
    x: Float[Tensor, "... d_model"],
    weight: Float[Tensor, "d_model"],
    eps: float = 1e-5
) -> Float[Tensor, "... d_model"]:
    """
    函数式 RMSNorm 实现
    
    参数:
        x: 输入张量 (..., d_model)
        weight: 缩放权重 (d_model,)
        eps: 数值稳定性参数
    
    返回:
        规范化后的张量 (..., d_model)
    """
    # 保存原始数据类型
    in_dtype = x.dtype
    
    # 转换到 float32 计算
    x = x.to(torch.float32)
    weight = weight.to(torch.float32)
    
    # 计算 RMS
    rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    
    # RMSNorm 变换
    result = x / rms * weight
    
    # 恢复原始数据类型
    return result.to(in_dtype)

# 使用示例和对比
if __name__ == "__main__":
    # 创建测试数据
    batch_size, seq_len, d_model = 2, 8, 512
    x = torch.randn(batch_size, seq_len, d_model)
    
    # 使用 RMSNorm
    rmsnorm = RMSNorm(d_model)
    output = rmsnorm(x)
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"输入统计: mean={x.mean():.4f}, std={x.std():.4f}")
    print(f"输出统计: mean={output.mean():.4f}, std={output.std():.4f}")
    
    # 验证 RMS 计算
    rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-5)
    manual_output = x / rms * rmsnorm.weight
    print(f"手动计算与模块输出是否一致: {torch.allclose(output, manual_output)}")