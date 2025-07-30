import torch
from torch import nn
import math

class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Positional Embedding (RoPE)
    
    RoPE 是一种新型的位置编码方法，通过旋转矩阵将位置信息编码到特征向量中。
    相比传统的绝对位置编码，RoPE 具有更好的相对位置建模能力和外推性。
    
    核心思想：
    - 将特征向量看作复数，通过旋转角度来编码位置信息
    - 每个位置对应不同的旋转角度
    - 不同特征维度使用不同的旋转频率
    
    数学原理：
    对于位置 m 和特征维度 i，旋转角度为：θ_i * m
    其中 θ_i = θ^(-2i/d_k)，θ 是基础频率参数
    """
    
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        """
        初始化 RoPE 模块
        
        参数:
            theta (float): 基础角度参数，控制旋转频率的基数
                          通常设为 10000.0，类似于 Transformer 原始位置编码
                          较大的 theta 意味着较低的基础频率
            d_k (int): 特征维度大小，必须是偶数
                       因为 RoPE 将相邻的特征对 (x_i, x_{i+1}) 看作复数进行旋转
            max_seq_len (int): 支持的最大序列长度
                              预计算这个长度内所有位置的 cos/sin 值
            device: 张量存储设备 (cpu/cuda)
        """
        super().__init__()
        
        # 确保 d_k 是偶数，因为我们将特征维度成对处理
        assert d_k % 2 == 0, f"d_k must be even, got {d_k}"
        
        # 1. 计算每个特征维度对的基础频率
        # torch.arange(0, d_k, 2): [0, 2, 4, ..., d_k-2] - 偶数索引
        # 形状: [d_k // 2]
        # 这些索引对应特征对的索引 (0,1), (2,3), (4,5), ...
        indices = torch.arange(0, d_k, 2, device=device, dtype=torch.float)
        
        # 计算 θ_i = θ^(-2i/d_k)，每个特征对的基础旋转频率
        # 越靠后的特征维度，旋转频率越低（角度变化越慢）
        # 形状: [d_k // 2]
        theta_ik = theta ** (-indices / d_k)
        
        # 2. 计算所有位置的位置索引
        # pos: [0, 1, 2, ..., max_seq_len-1]
        # 形状: [max_seq_len]
        pos = torch.arange(max_seq_len, device=device, dtype=torch.float)
        
        # 3. 计算所有 (位置, 特征维度对) 的旋转角度
        # einsum "i,j->ij": pos[i] * theta_ik[j]
        # 结果: angles[m][i] = m * θ_i，表示位置 m 在第 i 个特征对上的旋转角度
        # 形状: [max_seq_len, d_k // 2]
        angles = torch.einsum("i,j->ij", pos, theta_ik)
        
        # 4. 预计算所有角度的 cos 和 sin 值
        # 这些是旋转矩阵的基础元素
        # 使用 register_buffer 将它们注册为模型的非参数张量
        # persistent=False 表示在保存模型时不包含这些缓冲区（可以重新计算）
        self.register_buffer("cos", torch.cos(angles), persistent=False)
        self.register_buffer("sin", torch.sin(angles), persistent=False)
        
        # 存储维度信息用于调试
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.theta = theta

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        对输入张量应用 RoPE 位置编码
        
        参数:
            x: 输入张量，形状为 [..., d_k]
               通常是 (batch_size, num_heads, seq_len, d_k) 或 (batch_size, seq_len, d_k)
               最后一个维度必须等于 d_k
            token_positions: 位置索引张量，形状与 x 的前几个维度兼容
                           包含每个 token 在序列中的位置 (0, 1, 2, ...)
                           例如: [0, 1, 2, 3] 表示序列中的第0到第3个位置
        
        返回:
            应用 RoPE 后的张量，形状与输入 x 相同 [..., d_k]
        
        RoPE 变换过程:
        1. 将特征向量按维度分成实部和虚部 (x1, x2)
        2. 根据位置获取对应的旋转角度的 cos 和 sin
        3. 应用 2D 旋转矩阵进行旋转
        4. 重新组合成原始形状
        """
        
        # 1. 根据位置索引获取对应的 cos 和 sin 值
        # cos/sin: [max_seq_len, d_k // 2]
        # token_positions: [...] - 任意形状的位置索引
        # 结果: [..., d_k // 2] - 每个位置对应的旋转参数
        cos = self.cos[token_positions]  # 形状: [..., d_k // 2]
        sin = self.sin[token_positions]  # 形状: [..., d_k // 2]
        
        # 2. 将输入特征分解为相邻的特征对
        # 把特征维度看作复数：(x_0, x_1), (x_2, x_3), ..., (x_{d_k-2}, x_{d_k-1})
        # x1: 偶数索引特征 [x_0, x_2, x_4, ...] - 复数的实部
        # x2: 奇数索引特征 [x_1, x_3, x_5, ...] - 复数的虚部
        x1 = x[..., 0::2]  # 形状: [..., d_k // 2] - 从索引0开始，每隔2个取一个
        x2 = x[..., 1::2]  # 形状: [..., d_k // 2] - 从索引1开始，每隔2个取一个
        
        # 3. 应用 2D 旋转矩阵
        # 旋转矩阵: [[cos, -sin],
        #           [sin,  cos]]
        # 对复数 z = x1 + i*x2 进行旋转: z' = z * e^(iθ) = z * (cos + i*sin)
        # 实部: x1' = x1*cos - x2*sin
        # 虚部: x2' = x1*sin + x2*cos
        rotated_x1 = x1 * cos - x2 * sin  # 新的实部
        rotated_x2 = x1 * sin + x2 * cos  # 新的虚部
        
        # 4. 重新组合特征维度
        # torch.stack(..., dim=-1): 将 rotated_x1 和 rotated_x2 在最后一个维度堆叠
        # 形状变化: [..., d_k // 2] -> [..., d_k // 2, 2]
        # flatten(-2): 将最后两个维度展平
        # 形状变化: [..., d_k // 2, 2] -> [..., d_k]
        # 最终结果: [rotated_x1[0], rotated_x2[0], rotated_x1[1], rotated_x2[1], ...]
        out = torch.stack([rotated_x1, rotated_x2], dim=-1).flatten(-2)
        
        return out

# 辅助函数：用于理解 RoPE 的工作原理
def visualize_rope_angles(theta: float = 10000.0, d_k: int = 8, max_seq_len: int = 10):
    """
    可视化 RoPE 的角度分布，帮助理解其工作原理
    """
    print(f"RoPE 参数: θ={theta}, d_k={d_k}, max_seq_len={max_seq_len}")
    print("=" * 50)
    
    # 计算基础频率
    indices = torch.arange(0, d_k, 2, dtype=torch.float)
    theta_ik = theta ** (-indices / d_k)
    
    print("各特征对的基础频率 θ_i:")
    for i, freq in enumerate(theta_ik):
        print(f"  特征对 {i} (维度 {2*i}, {2*i+1}): θ_{i} = {freq:.6f}")
    
    print("\n前几个位置的旋转角度 (弧度):")
    pos = torch.arange(min(5, max_seq_len), dtype=torch.float)
    angles = torch.einsum("i,j->ij", pos, theta_ik)
    
    for m in range(len(pos)):
        print(f"  位置 {m}: {angles[m].tolist()}")
    
    print("\n观察:")
    print("1. 低频特征对（后面的维度）角度变化慢，适合捕捉长距离依赖")
    print("2. 高频特征对（前面的维度）角度变化快，适合捕捉短距离依赖")
    print("3. 不同位置具有唯一的角度组合，实现位置区分")

def run_rope(
    theta: float,
    d_k: int,
    x: torch.Tensor,
    token_positions: torch.Tensor
) -> torch.Tensor:
    """
    函数式 RoPE 实现，用于测试和理解
    
    参数:
        theta: 基础角度参数
        d_k: 特征维度
        x: 输入张量 [..., d_k]
        token_positions: 位置索引 [...]
    
    返回:
        应用 RoPE 后的张量 [..., d_k]
    """
    # 计算角度
    indices = torch.arange(0, d_k, 2, device=x.device, dtype=x.dtype)
    theta_ik = theta ** (-indices / d_k)
    angles = token_positions.unsqueeze(-1) * theta_ik.unsqueeze(0)
    
    # 计算旋转参数
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    
    # 分解特征
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    
    # 旋转
    rotated_x1 = x1 * cos - x2 * sin
    rotated_x2 = x1 * sin + x2 * cos
    
    # 重组
    return torch.stack([rotated_x1, rotated_x2], dim=-1).flatten(-2)

# 使用示例
if __name__ == "__main__":
    # 创建 RoPE 模块
    rope = RotaryPositionalEmbedding(theta=10000.0, d_k=8, max_seq_len=100)
    
    # 测试数据
    batch_size, seq_len, d_k = 2, 4, 8
    x = torch.randn(batch_size, seq_len, d_k)
    positions = torch.arange(seq_len)  # [0, 1, 2, 3]
    
    # 应用 RoPE
    output = rope(x, positions)
    
    print(f"输入形状: {x.shape}")
    print(f"位置索引: {positions}")
    print(f"输出形状: {output.shape}")
    print(f"应用 RoPE 前后形状是否一致: {x.shape == output.shape}")
    
    # 可视化角度分布
    print("\n" + "="*50)
    visualize_rope_angles()