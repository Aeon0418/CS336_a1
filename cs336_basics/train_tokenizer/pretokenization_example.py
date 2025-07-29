import os
from typing import BinaryIO


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    将文件分块，使每个部分可以独立计数。
    如果边界最终重叠，可能返回更少的块数。
    输出
    一个升序的整数列表 chunk_boundaries，表示各块的字节偏移，
    列表长度 = 块数 + 1，chunk_boundaries[0]=0，最后一项是文件总大小。
    """
    # 确保特殊token以字节字符串形式表示
    assert isinstance(split_special_token, bytes), "必须将特殊token表示为字节字符串"

    # 获取文件总大小（字节数）
    file.seek(0, os.SEEK_END)  # 移动到文件末尾
    file_size = file.tell()    # 获取当前位置（即文件大小）
    file.seek(0)               # 回到文件开头

    # 计算每个块的理论大小
    chunk_size = file_size // desired_num_chunks

    # 初始块边界位置的猜测，均匀分布
    # 块从前一个索引开始，不包括最后一个索引
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size  # 最后一个边界设为文件末尾

    # 每次向前读取的小块大小
    mini_chunk_size = 4096  # 一次向前读取4k字节

    # 调整中间的边界位置（不包括第一个和最后一个）
    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]  # 当前边界的初始位置
        file.seek(initial_position)  # 移动到边界猜测位置
        
        while True:
            # 读取一个小块
            mini_chunk = file.read(mini_chunk_size)

            # 如果到达文件末尾，将此边界设为文件结尾
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # 在小块中查找特殊token
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                # 找到特殊token，更新边界位置
                chunk_boundaries[bi] = initial_position + found_at
                break
            
            # 没有找到，继续向前搜索
            initial_position += mini_chunk_size

    # 确保所有边界都是唯一的，但可能少于期望的块数
    return sorted(set(chunk_boundaries))


## 使用示例
with open(..., "rb") as f:
    num_processes = 4  # 进程数
    # 找到基于特殊token的分块边界
    boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

    # 以下是串行实现，但你可以通过将每个起始/结束对
    # 发送给一组进程来并行化这个过程
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        f.seek(start)  # 移动到块的开始位置
        # 读取块内容并解码为UTF-8字符串
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        # 对你的块运行预分词，并存储每个预token的计数