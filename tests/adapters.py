from __future__ import annotations

import numpy as np
import os
from typing import IO, Any, BinaryIO
from collections.abc import Iterable
from jaxtyping import Float, Int

import math

import numpy.typing as npt
import torch
from torch import Tensor
import torch.nn as nn
# 这个文件是 CS336 作业的测试适配器，定义了所有需要实现的深度学习组件的接口
# 包括线性层、嵌入层、SwiGLU、缩放点积注意力、多头自注意力、RoPE、Transformer块等。
# 你需要实现这些函数以通过测试。

from cs336_basics.train_tokenizer.train_tokenizer2 import  train_bpe, merge_token_sequence
from cs336_basics.model.rmsnorm import RMSNorm
from cs336_basics.model.linear import Linear
from cs336_basics.model.embedding import Embedding
from cs336_basics.model.swiglu import SwiGLU
from cs336_basics.model.rope import RotaryPositionalEmbedding



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


def run_embedding(
    vocab_size: int,
    d_model: int,
    weights: Float[Tensor, " vocab_size d_model"],
    token_ids: Int[Tensor, " ..."],
) -> Float[Tensor, " ... d_model"]:
    """
    给定嵌入层的权重，获取一批token ID的嵌入。

    参数:
        vocab_size (int): 词汇表中嵌入的数量
        d_model (int): 嵌入维度的大小
        weights (Float[Tensor, "vocab_size d_model"]): 要获取的嵌入向量
        token_ids (Int[Tensor, "..."]): 要从嵌入层获取的token ID集合

    返回:
        Float[Tensor, "... d_model"]: 嵌入层返回的嵌入批次。
    """
    embedding = Embedding(vocab_size, d_model)
    with torch.no_grad():
        embedding.weight.copy_(weights)
    return embedding(token_ids)


def run_swiglu(
    d_model: int,
    d_ff: int,
    w1_weight: Float[Tensor, " d_ff d_model"],
    w2_weight: Float[Tensor, " d_model d_ff"],
    w3_weight: Float[Tensor, " d_ff d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """
    给定SwiGLU网络的权重，返回使用这些权重的实现输出。

    参数:
        d_model (int): 前馈输入和输出的维度。
        d_ff (int): SwiGLU内部上投影的维度。
        w1_weight (Float[Tensor, "d_ff d_model"]): W1的存储权重
        w2_weight (Float[Tensor, "d_model d_ff"]): W2的存储权重
        w3_weight (Float[Tensor, "d_ff d_model"]): W3的存储权重
        in_features (Float[Tensor, "... d_model"]): 前馈层的输入嵌入。

    返回:
        Float[Tensor, "... d_model"]: 与输入嵌入形状相同的输出嵌入。
    """
    
    # 示例:
    # 如果你的状态字典键匹配，可以使用 `load_state_dict()`
    #swiglu.load_state_dict(weights)
    # 你也可以手动分配权重
    # swiglu.w1.weight.data = w1_weight
    # swiglu.w2.weight.data = w2_weight
    # swiglu.w3.weight.data = w3_weight
    swiglu = SwiGLU(d_model, d_ff)
    swiglu.load_weights(w1_weight, w2_weight, w3_weight)
    # with torch.no_grad():
    #     swiglu.w1.weight.copy_(w1_weight_adj.T)
    #     swiglu.w2.weight.copy_(w2_weight_adj.T)
    #     swiglu.w3.weight.copy_(w3_weight_adj.T)

    return swiglu(in_features)


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




def run_multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
) -> Float[Tensor, " ... sequence_length d_out"]:
    """
    给定多头注意力朴素非批次实现的键、查询和值投影权重，
    返回优化批次实现的输出。此实现应在单个矩阵乘法中处理
    所有头的键、查询和值投影。
    此函数不应使用RoPE。
    参见Vaswani等人，2017年的第3.2.2节。

    参数:
        d_model (int): 前馈输入和输出的维度。
        num_heads (int): 多头注意力中使用的头数。
        q_proj_weight (Float[Tensor, "d_k d_in"]): Q投影的权重
        k_proj_weight (Float[Tensor, "d_k d_in"]): K投影的权重
        v_proj_weight (Float[Tensor, "d_k d_in"]): V投影的权重
        o_proj_weight (Float[Tensor, "d_model d_v"]): 输出投影的权重
        in_features (Float[Tensor, "... sequence_length d_in"]): 运行实现的张量。

    返回:
        Float[Tensor, " ... sequence_length d_out"]: 使用给定QKV投影权重和输入特征
        运行优化批次多头注意力实现的输出张量。
    """
    from cs336_basics.model.msa import Multihead_self_attention
    multihead_att = Multihead_self_attention(d_model, num_heads)
    with torch.no_grad():
        multihead_att.q_proj.weight.copy_(q_proj_weight.T)
        multihead_att.k_proj.weight.copy_(k_proj_weight.T)
        multihead_att.v_proj.weight.copy_(v_proj_weight.T)
        multihead_att.o_proj.weight.copy_(o_proj_weight.T)
    return multihead_att(in_features)


def run_multihead_self_attention_with_rope(
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
    """
    给定多头注意力朴素非批次实现的键、查询和值投影权重，
    返回优化批次实现的输出。此实现应在单个矩阵乘法中处理
    所有头的键、查询和值投影。
    此版本的MHA应包含RoPE。
    在这种情况下，RoPE嵌入维度必须是头嵌入维度(d_model // num_heads)。
    参见Vaswani等人，2017年的第3.2.2节。

    参数:
        d_model (int): 前馈输入和输出的维度。
        num_heads (int): 多头注意力中使用的头数。
        max_seq_len (int): 如果你的实现预缓存的最大序列长度。
        theta (float): RoPE参数。
        q_proj_weight (Float[Tensor, "d_k d_in"]): Q投影的权重
        k_proj_weight (Float[Tensor, "d_k d_in"]): K投影的权重
        v_proj_weight (Float[Tensor, "d_k d_in"]): V投影的权重
        o_proj_weight (Float[Tensor, "d_model d_v"]): 输出投影的权重
        in_features (Float[Tensor, "... sequence_length d_in"]): 运行实现的张量。
        token_positions (Int[Tensor, " ... sequence_length"] | None): 带有token位置的可选张量

    返回:
        Float[Tensor, " ... sequence_length d_out"]: 使用给定QKV投影权重和输入特征
        运行优化批次多头注意力实现的输出张量。
    """
    from cs336_basics.model.msa import Multihead_self_attention
    from cs336_basics.model.rope import RotaryPositionalEmbedding
    rope = RotaryPositionalEmbedding(theta, d_model // num_heads, max_seq_len)
    
    # 创建多头注意力实例，只传递必要的参数
    multihead_att = Multihead_self_attention(d_model, num_heads, rope)
    with torch.no_grad():
        multihead_att.q_proj.weight.copy_(q_proj_weight.T)
        multihead_att.k_proj.weight.copy_(k_proj_weight.T)
        multihead_att.v_proj.weight.copy_(v_proj_weight.T)
        multihead_att.o_proj.weight.copy_(o_proj_weight.T)
    return multihead_att(in_features, token_positions)


def run_rope(
    d_k: int,
    theta: float,
    max_seq_len: int,
    in_query_or_key: Float[Tensor, " ... sequence_length d_k"],
    token_positions: Int[Tensor, " ... sequence_length"],
) -> Float[Tensor, " ... sequence_length d_k"]:
    """
    为给定输入张量运行RoPE。

    参数:
        d_k (int): 查询或键张量的嵌入维度大小。
        theta (float): RoPE参数。
        max_seq_len (int): 如果你的实现预缓存的最大序列长度。
        in_query_or_key (Float[Tensor, "... sequence_length d_k"]): 运行RoPE的输入张量。
        token_positions (Int[Tensor, "... sequence_length"]): 形状为(batch_size, sequence_length)的张量，包含token位置
    返回:
        Float[Tensor, " ... sequence_length d_k"]: 应用RoPE的输入张量。
    """
    
    rope = RotaryPositionalEmbedding(theta, d_k, max_seq_len)
    return rope(in_query_or_key, token_positions)


def run_transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Tensor],
    in_features: Float[Tensor, " batch sequence_length d_model"],
) -> Float[Tensor, " batch sequence_length d_model"]:
    """
    给定预归一化Transformer块的权重和输入特征，
    返回在输入特征上运行Transformer块的输出。

    此函数应使用RoPE。
    根据你的实现，你可能只需要将相关参数传递给
    TransformerBlock构造函数，或者你可能需要初始化自己的RoPE
    类并传递它。

    参数:
        d_model (int): Transformer块输入的维度。
        num_heads (int): 多头注意力中使用的头数。`d_model`必须
            能被`num_heads`整除。
        d_ff (int): 前馈内层的维度。
        max_seq_len (int): 如果你的实现预缓存的最大序列长度。
        theta (float): RoPE参数。
        weights (dict[str, Tensor]):
            我们参考实现的状态字典。
            此字典的键包括:
            - `attn.q_proj.weight`
                所有`num_heads`个注意力头的查询投影。
                形状为(d_model, d_model)。
                行按形状为(num_heads, d_k)的矩阵排序，
                所以`attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`。
            - `attn.k_proj.weight`
                所有`num_heads`个注意力头的键投影。
                形状为(d_model, d_model)。
                行按形状为(num_heads, d_k)的矩阵排序，
                所以`attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`。
            - `attn.v_proj.weight`
                所有`num_heads`个注意力头的值投影。
                形状为(d_model, d_model)。
                行按形状为(num_heads, d_v)的矩阵排序，
                所以`attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`。
            - `attn.output_proj.weight`
                多头自注意力输出投影的权重
                形状为(d_model, d_model)。
            - `ln1.weight`
                变换器块中应用的第一个RMSNorm的仿射变换权重。
                形状为(d_model,)。
            - `ffn.w1.weight`
                FFN中第一个线性变换的权重。
                形状为(d_model, d_ff)。
            - `ffn.w2.weight`
                FFN中第二个线性变换的权重。
                形状为(d_ff, d_model)。
            - `ffn.w3.weight`
                FFN中第三个线性变换的权重。
                形状为(d_model, d_ff)。
            - `ln2.weight`
                变换器块中应用的第二个RMSNorm的仿射变换权重。
                形状为(d_model,)。
        in_features (Float[Tensor, "batch sequence_length d_model"]):
            运行实现的张量。

    返回:
        Float[Tensor, "batch sequence_length d_model"] 在使用RoPE时
        在输入特征上运行Transformer块的输出张量。
    """
    raise NotImplementedError


def run_transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Tensor],
    in_indices: Int[Tensor, " batch_size sequence_length"],
) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
    """
    给定Transformer语言模型的权重和输入索引，
    返回在输入索引上运行前向传播的输出。

    此函数应使用RoPE。

    参数:
        vocab_size (int): 要预测的输出词汇表中的唯一项目数。
        context_length (int): 一次处理的最大token数。
        d_model (int): 模型嵌入和子层输出的维度。
        num_layers (int): 要使用的Transformer层数。
        num_heads (int): 多头注意力中使用的头数。`d_model`必须
            能被`num_heads`整除。
        d_ff (int): 前馈内层的维度(第3.3节)。
        rope_theta (float): RoPE Θ参数。
        weights (dict[str, Tensor]):
            我们参考实现的状态字典。{num_layers}指的是
            `0`到`num_layers - 1`之间的整数(层索引)。
            此字典的键包括:
            - `token_embeddings.weight`
                Token嵌入矩阵。形状为(vocab_size, d_model)。
            - `layers.{num_layers}.attn.q_proj.weight`
                所有`num_heads`个注意力头的查询投影。
                形状为(num_heads * (d_model / num_heads), d_model)。
                行按形状为(num_heads, d_k)的矩阵排序，
                所以`attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`。
            - `layers.{num_layers}.attn.k_proj.weight`
                所有`num_heads`个注意力头的键投影。
                形状为(num_heads * (d_model / num_heads), d_model)。
                行按形状为(num_heads, d_k)的矩阵排序，
                所以`attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`。
            - `layers.{num_layers}.attn.v_proj.weight`
                所有`num_heads`个注意力头的值投影。
                形状为(num_heads * (d_model / num_heads), d_model)。
                行按形状为(num_heads, d_v)的矩阵排序，
                所以`attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`。
            - `layers.{num_layers}.attn.output_proj.weight`
                多头自注意力输出投影的权重
                形状为((d_model / num_heads) * num_heads, d_model)。
            - `layers.{num_layers}.ln1.weight`
                变换器块中应用的第一个RMSNorm的仿射变换权重。
                形状为(d_model,)。
            - `layers.{num_layers}.ffn.w1.weight`
                FFN中第一个线性变换的权重。
                形状为(d_model, d_ff)。
            - `layers.{num_layers}.ffn.w2.weight`
                FFN中第二个线性变换的权重。
                形状为(d_ff, d_model)。
            - `layers.{num_layers}.ffn.w3.weight`
                FFN中第三个线性变换的权重。
                形状为(d_model, d_ff)。
            - `layers.{num_layers}.ln2.weight`
                变换器块中应用的第二个RMSNorm的仿射变换权重。
                形状为(d_model,)。
            - `ln_final.weight`
                应用于最终变换器块输出的RMSNorm的仿射变换权重。
                形状为(d_model, )。
            - `lm_head.weight`
                语言模型输出嵌入的权重。
                形状为(vocab_size, d_model)。
        in_indices (Int[Tensor, "batch_size sequence_length"]) 运行语言模型的输入索引张量。形状为(batch_size, sequence_length)，其中
            `sequence_length`最多为`context_length`。

    返回:
        Float[Tensor, "batch_size sequence_length vocab_size"]: 每个token的预测未归一化
        下一词分布的张量。
    """
    raise NotImplementedError


def run_rmsnorm(
    d_model: int,
    eps: float,
    weights: Float[Tensor, " d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """
    给定RMSNorm仿射变换的权重，
    返回在输入特征上运行RMSNorm的输出。

    参数:
        d_model (int): RMSNorm输入的维度。
        eps: (float): 为数值稳定性添加到分母的值。
        weights (Float[Tensor, "d_model"]): RMSNorm权重。
        in_features (Float[Tensor, "... d_model"]): 运行RMSNorm的输入特征。可以有任意的前导
            维度。

    返回:
        Float[Tensor,"... d_model"]: 与`in_features`形状相同的张量，包含在
        `in_features`上运行RMSNorm的输出。
    """
    
    rmsnorm = RMSNorm(d_model, eps)
    with torch.no_grad():
        rmsnorm.weight.copy_(weights)
    return rmsnorm(in_features)


def run_silu(in_features: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
    """
    给定输入张量，返回对每个元素应用SiLU的输出。

    参数:
        in_features(Float[Tensor, "..."]): 运行SiLU的输入特征。形状任意。

    返回:
        Float[Tensor,"..."]: 与`in_features`形状相同的张量，包含对每个元素应用
        SiLU的输出。
    """
    # 最后一维做哈达玛积
    return in_features * torch.sigmoid(in_features)


def run_get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    给定数据集(整数的一维numpy数组)和期望的批量大小及
    上下文长度，从数据集中采样语言建模输入序列和相应的标签。

    参数:
        dataset (np.array): 数据集中整数token ID的一维numpy数组。
        batch_size (int): 期望采样的批量大小。
        context_length (int): 每个采样样本的期望上下文长度。
        device (str): PyTorch设备字符串(例如，'cpu'或'cuda:0')，指示放置
            采样输入序列和标签的设备。

    返回:
        形状为(batch_size, context_length)的torch.LongTensor元组。第一个元组项
        是采样的输入序列，第二个元组项是相应的语言建模标签。
    """
    st = torch.randint(len(dataset) - context_length, (batch_size,))
    input_seq = torch.stack([torch.from_numpy((dataset[i : i + context_length]).astype(np.int64)) for i in st])
    target_seq = torch.stack([torch.from_numpy((dataset[i + 1 : i + 1 + context_length]).astype(np.int64)) for i in st])
    if 'cuda' in device:
        input_seq = input_seq.pin_memory().to(device, non_blocking=True)
        target_seq = target_seq.pin_memory().to(device, non_blocking=True)
    else:
        input_seq = input_seq.to(device)
        target_seq = target_seq.to(device)
    return input_seq, target_seq


def run_softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    """
    给定输入张量，返回对输入的给定`dim`应用softmax的输出。

    参数:
        in_features (Float[Tensor, "..."]): 要softmax的输入特征。形状任意。
        dim (int): 要应用softmax的`in_features`的维度。

    返回:
        Float[Tensor, "..."]: 与`in_features`形状相同的张量，包含对指定`dim`
        进行softmax归一化的输出。
    """
    dim_max = torch.amax(in_features, dim=dim, keepdim=True)
    dim_exp = torch.exp(in_features - dim_max)
    sum_dim_exp = torch.sum(dim_exp, dim=dim, keepdim=True)
    return dim_exp / sum_dim_exp


def run_cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:
    """
    给定输入和目标张量，计算样本间的平均交叉熵损失。

    参数:
        inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j]是
            第i个样本的第j个类的未归一化logit。
        targets (Int[Tensor, "batch_size"]): 形状为(batch_size,)的张量，包含正确类的索引。
            每个值必须在0和`num_classes - 1`之间。

    返回:
        Float[Tensor, ""]: 样本间的平均交叉熵损失。
    """
    dim_max = torch.amax(inputs, dim=-1, keepdim=True)
    dim_submax = inputs - dim_max
    dim_logsumexp = dim_submax - torch.log(torch.sum(torch.exp(dim_submax), dim=-1, keepdim=True))
    return torch.mean(torch.gather(input=-dim_logsumexp, dim=-1, index=targets.unsqueeze(-1)))



def run_gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """
    给定一组参数，裁剪它们的组合梯度，使l2范数最多为max_l2_norm。

    参数:
        parameters (Iterable[torch.nn.Parameter]): 可训练参数的集合。
        max_l2_norm (float): 包含最大l2范数的正值。

    参数的梯度(parameter.grad)应就地修改。
    """
    grads = []
    for pt in parameters:
        if pt.grad is not None:
            grads.append(pt.grad)
    grads_l2norm = 0.0
    for gd in grads:
        grads_l2norm += (gd ** 2).sum()
    grads_l2norm = torch.sqrt(grads_l2norm)
    if grads_l2norm >= max_l2_norm:
        ft = max_l2_norm / (grads_l2norm + 1e-6)
        for gd in grads:
            gd *= ft


def get_adamw_cls() -> type[torch.optim.Optimizer]:
    """
    返回实现AdamW的torch.optim.Optimizer。
    """
    from cs336_basics.model.adamw import AdamW  # 假设有一个AdamW实现
    return AdamW


def run_get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    """
    给定余弦学习率衰减调度(带线性预热)的参数和迭代次数，
    返回在指定调度下给定迭代的学习率。

    参数:
        it (int): 获取学习率的迭代次数。
        max_learning_rate (float): alpha_max，余弦学习率调度(带预热)的最大学习率。
        min_learning_rate (float): alpha_min，余弦学习率调度(带预热)的最小/最终学习率。
        warmup_iters (int): T_w，线性预热学习率的迭代次数。
        cosine_cycle_iters (int): T_c，余弦退火迭代次数。

    返回:
        在指定调度下给定迭代的学习率。
    """
    alpha_t = 0.0
    if it < warmup_iters:
        alpha_t = it / warmup_iters * max_learning_rate
    elif it >= warmup_iters and it <= cosine_cycle_iters:
        alpha_t = min_learning_rate + 0.5 * (1 + math.cos(((it-warmup_iters)/(cosine_cycle_iters-warmup_iters))*math.pi)) * (max_learning_rate - min_learning_rate)
    elif it > cosine_cycle_iters:
        alpha_t = min_learning_rate
    return alpha_t


def run_save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    """
    给定模型、优化器和迭代次数，将它们序列化到磁盘。

    参数:
        model (torch.nn.Module): 序列化此模型的状态。
        optimizer (torch.optim.Optimizer): 序列化此优化器的状态。
        iteration (int): 序列化此值，它表示我们已完成的训练迭代次数。
        out (str | os.PathLike | BinaryIO | IO[bytes]): 将模型、优化器和迭代序列化到的路径或类文件对象。
    """
    checkpoints = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "iteration": iteration
    }
    torch.save(checkpoints, out)


def run_load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
):
    """
    给定序列化检查点(路径或类文件对象)，将序列化状态恢复到给定模型和优化器。
    返回我们之前在检查点中序列化的迭代次数。

    参数:
        src (str | os.PathLike | BinaryIO | IO[bytes]): 序列化检查点的路径或类文件对象。
        model (torch.nn.Module): 恢复此模型的状态。
        optimizer (torch.optim.Optimizer): 恢复此优化器的状态。
    返回:
        int: 之前序列化的迭代次数。
    """
    checkpoints = torch.load(src)
    model.load_state_dict(checkpoints["model_state"])
    optimizer.load_state_dict(checkpoints["optimizer_state"])
    return checkpoints["iteration"]


def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> Any:
    """
    给定词汇表、合并列表和特殊token列表，
    返回使用提供的词汇表、合并和特殊token的BPE分词器。

    参数:
        vocab (dict[int, bytes]): 分词器词汇表，从int(词汇表中的token ID)
            到bytes(token字节)的映射
        merges (list[tuple[bytes, bytes]]): BPE合并。每个列表项是字节元组(<token1>, <token2>)，
            表示<token1>与<token2>合并。
            合并按创建顺序排序。
        special_tokens (list[str] | None): 分词器的字符串特殊token列表。这些字符串永远不会
            被分割成多个token，并且总是保持为单个token。

    返回:
        使用提供的词汇表、合并和特殊token的BPE分词器。
    """
    from cs336_basics.model.get_tokenizer import Tokenizer  # 假设有一个BPE分词器实现
    tokenizer = Tokenizer(vocab, merges, special_tokens)
    return tokenizer






def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    给定输入语料库的路径，训练BPE分词器并输出其词汇表和合并。

    参数:
        input_path (str | os.PathLike): BPE分词器训练数据的路径。
        vocab_size (int): 分词器词汇表中的项目总数(包括特殊token)。
        special_tokens (list[str]): 要添加到分词器词汇表的字符串特殊token列表。
            这些字符串永远不会被分割成多个token，并且总是
            保持为单个token。如果这些特殊token出现在`input_path`中，
            它们被视为任何其他字符串。

    返回:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                训练的分词器词汇表，从int(词汇表中的token ID)
                到bytes(token字节)的映射
            merges:
                BPE合并。每个列表项是字节元组(<token1>, <token2>)，
                表示<token1>与<token2>合并。
                合并按创建顺序排序。
    """
    vocab, merges = train_bpe(
        input_path=input_path,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        **kwargs
    )

    return vocab, merges # 返回最终的词汇表和合并记录
