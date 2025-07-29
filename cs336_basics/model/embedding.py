import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float, Int  #用来给函数签名做更精确的张量形状/类型注解


class Embedding(nn.Module):
    '''
    数字到 d_model的查表映射
    '''
    def __init__(self,num_embedding,embedding_dim,device=None,dtype=None):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}

        # 创建空的权重矩阵，从vocab_size到d_model
        weight = torch.empty(num_embedding,embedding_dim,**factory_kwargs)

        torch.nn.init.trunc_normal_(weight, mean=0.0, std=1.0, a=-3, b=3)

        #把这个张量包装成可训练的参数，注册到模型里
        self.weight = nn.Parameter(weight)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        #token_ids：通常是一个整数张量（任意形状），每个元素是一个 token 的 ID
        #利用 PyTorch 的张量索引功能直接做查表（lookup），输出形状 [..., d_model] 的嵌入向量。
        out = self.weight[token_ids]
        return out



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