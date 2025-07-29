from tqdm import tqdm
import os
import math
import wandb  # 权重和偏差追踪工具，用于实验监控
import argparse
import torch
import time
import pickle
import pathlib
import numpy as np
from collections.abc import Iterable
from torch import Tensor
from jaxtyping import Float, Int
from typing import IO, Any, BinaryIO
import numpy.typing as npt

# 导入自定义模型组件
from cs336_basics.train_tokenizer.bpe_trainer import run_train_bpe
from cs336_basics.model.get_tokenizer import Tokenizer
from cs336_basics.model.transformer import Transformer_lm
from cs336_basics.model.adamw import AdamW

# 设置数据和模型路径
DATA_PATH = (pathlib.Path(__file__).resolve().parent.parent.parent) / "data"
#/home/aeon/Desktop/Learn/CS336/Assignment/assignment1-basics/data/
MODULE_PATH = (pathlib.Path(__file__).resolve().parent.parent.parent) / "chkpt"
#/home/aeon/Desktop/Learn/CS336/Assignment/assignment1-basics/module/



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
    包含内容：
        模型权重：所有Transformer层的参数
        优化器状态：AdamW的动量、梯度历史等
        训练进度：当前迭代次数
        Tsccheckpoint.pt：检查点文件名，用于断点重续

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
def save_pkl(file, file_name):
    """保存Python对象到pickle文件"""
    with open(file_name, 'wb') as f:
        pickle.dump(file, f)

def load_pkl(file_name):
    """从pickle文件加载Python对象"""
    with open(file_name, 'rb') as f:
        file = pickle.load(f)
        return file

def save_encode(file, file_name):
    """将token列表保存为二进制文件（uint16格式）"""
    np.array(file, dtype=np.uint16).tofile(file_name)

def save_encode_stream(token_stream: Iterable[int], file_path: os.PathLike):
    """流式保存token序列到二进制文件（内存友好）"""
    array = np.fromiter(token_stream, dtype=np.uint16)
    array.tofile(file_path)

def train_bpe_TinyStories(
    file_name: str | os.PathLike, 
    vocab_size: int, 
    special_tokens: list[str], 
    vocab_name: str, 
    merges_name: str
):
    """训练BPE分词器并保存词汇表和合并规则"""
    print("开始训练BPE分词器...")
    start_time = time.time()
    traindata_path = DATA_PATH / file_name
    
    # 训练BPE分词器
    vocab, merges = run_train_bpe(
        input_path=traindata_path,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
    )
    
    # 保存词汇表和合并规则
    save_pkl(vocab, DATA_PATH / vocab_name)
    save_pkl(merges, DATA_PATH / merges_name)
    
    # 计算训练时间
    end_time = time.time()
    duration = end_time - start_time
    minutes = int(duration // 60)
    seconds = int(duration % 60)
    print(f"BPE训练完成，执行时间: {minutes} 分 {seconds} 秒")

def Tokenizer_TinyStories(
    trainfile_name: str | os.PathLike, 
    validfile_name: str | os.PathLike, 
    trainencode_name: str | os.PathLike, 
    validencode_name: str | os.PathLike, 
    vocab_name: str | os.PathLike, 
    merges_name: str | os.PathLike, 
    special_tokens: list[str]
):
    """使用训练好的BPE分词器对数据集进行编码"""
    print("开始对数据集进行token化...")
    start_time = time.time()
    
    # 设置文件路径
    trainfile_path = DATA_PATH / trainfile_name
    validfile_path = DATA_PATH / validfile_name
    trainencode_path = DATA_PATH / trainencode_name
    validencode_path = DATA_PATH / validencode_name
    
    # 加载训练好的分词器
    tokenizer = Tokenizer.from_files(DATA_PATH / vocab_name, DATA_PATH / merges_name, special_tokens)

    # 处理训练集（流式编码以节省内存）
    with open(trainfile_path, 'r', encoding='utf-8') as f:
        train_lines = f.readlines()

    # 可选：计算压缩比和吞吐量的代码（已注释）
    # total_bytes = sum(len(line.encode('utf-8')) for line in train_lines)
    # encode_stream = tokenizer.encode_iterable(train_lines)
    # token_list = list(encode_stream)
    # total_tokens = len(token_list)
    # compression_ratio = total_bytes / total_tokens if total_tokens > 0 else float('inf')

    # 流式编码并保存训练集
    encode_stream = tokenizer.encode_iterable(train_lines)
    save_encode_stream(encode_stream, trainencode_path)
    print("训练集token化完成")

    # 处理验证集（代码已注释，可按需启用）
    with open(validfile_path, 'r', encoding='utf-8') as f:
        valid_lines = f.readlines()
    encode_stream = tokenizer.encode_iterable(valid_lines)
    save_encode_stream(encode_stream, validencode_path)

@torch.no_grad()
def evaluate_validloss(model, valid_dataset, batch_size, context_length, device):
    """评估模型在验证集上的损失"""
    model.eval()  # 切换到评估模式
    losses = []
    total_batches = len(valid_dataset) // (batch_size * context_length)

    # 遍历验证集批次
    for i in range(total_batches):
        input_batch, target_batch = run_get_batch(valid_dataset, batch_size, context_length, device)
        logits = model(input_batch)  # 前向传播
        loss = run_cross_entropy(logits.view(-1, logits.size(-1)), target_batch.view(-1))
        losses.append(loss.item())

    model.train()  # 恢复训练模式
    return sum(losses) / len(losses)  # 返回平均损失

def generate_sample_and_log(model, tokenizer, prompt_str, device, iteration, max_gen_tokens=256, temperature=1.0, top_p=0.95):
    """生成文本样本并记录到wandb"""
    model.eval()
    with torch.no_grad():
        # 编码输入提示
        prompt_ids = tokenizer.encode(prompt_str)
        input_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        eos_token_id = tokenizer.vocab_to_id.get("<|endoftext|>".encode('utf-8'), None)

        # 生成文本
        gen_ids = model.generate(
            input_tensor,
            max_gen_tokens=max_gen_tokens,
            temperature=temperature,  # 控制随机性
            top_p=top_p,             # nucleus采样参数
            eos_token_id=eos_token_id,
        )

        # 解码生成的文本
        full_ids = prompt_ids + gen_ids[0].tolist()
        output_text = tokenizer.decode(full_ids)

        # 打印和记录生成的文本
        print(f"[样本 @ 迭代 {iteration}] {output_text}")
        wandb.log({"sample/text": wandb.Html(f"<pre>{output_text}</pre>")})

    model.train()  # 恢复训练模式

if __name__ == '__main__':
    # ===== 超参数设置 =====
    trainfile_name = 'TinyStoriesV2-GPT4-train.txt'     # 训练数据文件
    validfile_name = 'TinyStoriesV2-GPT4-valid.txt'     # 验证数据文件
    vocab_name = 'vocab.pkl'                # 词汇表文件
    merges_name = 'merges.pkl'              # 合并规则文件
    trainencode_name = 'TStrain_tokens.bin'             # 编码后的训练数据
    validencode_name = 'TSvalid_tokens.bin'             # 编码后的验证数据
    
    # 模型超参数
    vocab_size = 10000          # 词汇表大小
    batch_size = 256            # 批次大小
    context_length = 256        # 上下文长度
    d_model = 256              # 模型维度
    d_ff = 1344                # 前馈网络维度
    initial_lr = 0.0033        # 初始学习率
    lr = 0.0033                # 当前学习率
    rope_theta = 10000         # RoPE的theta参数
    n_layers = 4               # Transformer层数
    n_heads = 16               # 注意力头数
    max_l2_norm = 1e-2         # 梯度裁剪的最大L2范数
    
    # 生成参数
    max_gen_tokens = 256       # 最大生成token数
    temperature = 0.8          # 生成温度
    top_p = 0.95              # nucleus采样参数
    special_tokens = ["<|endoftext|>"]  # 特殊token
    
    # ===== 数据预处理 =====
    # 训练BPE分词器（如果需要）
    # train_bpe_TinyStories(trainfile_name, vocab_size, special_tokens, vocab_name, merges_name)
    
    # 加载分词器
    tokenizer = Tokenizer.from_files(DATA_PATH / vocab_name, DATA_PATH / merges_name, special_tokens)
    
    # 对数据集进行token化（如果需要）
    # Tokenizer_TinyStories(trainfile_name, validfile_name, trainencode_name, validencode_name, vocab_name, merges_name, special_tokens)
    
    # ===== 设备设置 =====
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = "mps"
    print(f"使用设备: {device}")
    
    # ===== 数据加载 =====
    # 使用内存映射加载大型数据集（内存友好）
    train_dataset = np.memmap(DATA_PATH / trainencode_name, dtype=np.uint16, mode="r")
    valid_dataset = np.memmap(DATA_PATH / validencode_name, dtype=np.uint16, mode="r")
    
    # ===== 训练设置 =====
    start_iter = 0              # 起始迭代
    total_iters = 5000          # 总迭代次数
    log_interval = total_iters // 200    # 日志记录间隔
    ckpt_interval = total_iters // 20    # 检查点保存间隔
    val_interval = total_iters // 20     # 验证间隔
    print(f"总迭代次数: {total_iters}")
    
    # ===== 初始化wandb实验追踪 =====
    wandb.init(
        project="cs336_ass1",
        name=f"run-{time.strftime('%Y%m%d-%H%M%S')}",
        config={
            "batch_size": batch_size,
            "context_length": context_length,
            "max_lr": lr,
            "min_lr": max(1e-6, lr * 0.01),
            "warmup_iters": min(500, total_iters*0.1),
            "cosine_iters": total_iters,
        }
    )
    
    # ===== 模型初始化 =====
    model = Transformer_lm(
        vocab_size=vocab_size, 
        context_length=context_length, 
        num_layers=n_layers, 
        d_model=d_model, 
        num_heads=n_heads, 
        d_ff=d_ff, 
        rope_theta=rope_theta
    ).to(device)
    
    # ===== 优化器设置 =====
    # 使用AdamW优化器（默认参数）
    optimizer = AdamW(model.parameters(), lr=lr)
    
    # ===== 检查点恢复 =====
    ckpt_path = MODULE_PATH / 'TScheckpoint.pt'
    if ckpt_path.exists():
        start_iter = run_load_checkpoint(src=ckpt_path, model=model, optimizer=optimizer)
        print(f"从检查点恢复，起始迭代: {start_iter}")
    
    # ===== 训练准备 =====
    model.train()  # 设置为训练模式
    wandb.watch(model, log="all")  # wandb监控模型
    pbar = tqdm(total=total_iters)  # 进度条
    iteration = start_iter
    best_val_loss = float('inf')  # 最佳验证损失
    
    # ===== 训练循环 =====
    while iteration < total_iters:
        # 获取训练批次
        input_train, target_train = run_get_batch(train_dataset, batch_size, context_length, device)
        
        # 前向传播
        logits = model(input_train)
        loss = run_cross_entropy(logits.view(-1, logits.size(-1)), target_train.view(-1))
        
        # 学习率调度（余弦退火）
        lr = run_get_lr_cosine_schedule(
            iteration,
            max_learning_rate=initial_lr,
            min_learning_rate=max(1e-6, initial_lr * 0.01),
            warmup_iters=int(min(500, total_iters * 0.1)),
            cosine_cycle_iters=total_iters,
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        run_gradient_clipping(model.parameters(), max_l2_norm)  # 梯度裁剪
        optimizer.step()
        
        # ===== 日志记录 =====
        if iteration % log_interval == 0:
            print(f"[迭代 {iteration}] 损失: {loss.item():.4f}")
            wandb.log({"train/loss": loss.item(), "lr": lr}, step=iteration)
        
        # ===== 保存检查点 =====
        if iteration % ckpt_interval == 0:
            run_save_checkpoint(model, optimizer, iteration, ckpt_path)
        
        # ===== 验证评估 =====
        if iteration % val_interval == 0:
            val_loss = evaluate_validloss(model, valid_dataset, batch_size, context_length, device)
            print(f"[迭代 {iteration}] 验证损失: {val_loss:.4f}")
            wandb.log({"val/loss": val_loss}, step=iteration)
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                model.save_pretrained(MODULE_PATH)
                print(f"保存最佳模型 (验证损失={val_loss:.4f})")
                wandb.run.summary["best_val_loss"] = best_val_loss
            
            # 可选：生成文本样本（已注释）
            # generate_sample_and_log(model=model,
            #     tokenizer=tokenizer,
            #     prompt_str="从前有一天",  # 可以自定义提示词
            #     device=device,
            #     iteration=iteration,
            #     max_gen_tokens=max_gen_tokens,
            #     temperature=temperature,
            #     top_p=top_p,
            # )
        
        iteration += 1
        pbar.update(1)
    
    # ===== 训练结束 =====
    print("训练完成！")
    wandb.finish()
