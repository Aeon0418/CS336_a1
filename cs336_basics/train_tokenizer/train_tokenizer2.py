import os
import regex as re
from typing import Tuple, List, Dict, Set
from collections import defaultdict 
import pickle
import mmap
from tqdm import tqdm
from joblib import Parallel, delayed
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
    if not split_special_token:
        raise ValueError("必须提供至少一个特殊token")
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

def gpt2_pre_tokenize(text: str) -> list:
    """使用 GPT-2 的分词规则预分词文本。
    此函数根据 GPT-2 的预分词模式将文本分割成词符，处理以下情况：
    - 缩写（例如，'s, 'll, 've, 're）
    - 包含字母的单词
    - 数字
    - 标点符号和特殊字符
    - 空白字符
    参数:
        text (str): 需要预分词的输入文本
    返回:
        list: 基于 GPT-2 预分词规则的字符串词符列表
    示例:
        >>> gpt2_pre_tokenize("I'll go to NY!")
        ["I", "'ll", " go", " to", " NY", "!"]
    """

    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    return [m.group() for m in re.finditer(PAT, text)]


def tokenize_chunk(chunk_and_special_tokens: tuple[str, list[str]]) -> list[str]:
    chunk, special_tokens = chunk_and_special_tokens
    special_pat = "|".join(re.escape(tok) for tok in special_tokens)

    sub_chunks = [s for s in re.split(f"({special_pat})", chunk) if s]
    all_tokens = []
    for sub_chunk in sub_chunks:
        if sub_chunk in special_tokens:
            all_tokens.append(sub_chunk)
        else:
            all_tokens.extend(gpt2_pre_tokenize(sub_chunk))
    # sub_chunks = [s for s in re.split(special_pat, chunk) if s]
    # # 对每个子块进行预分词
    # all_tokens = []
    # for sub_chunk in sub_chunks:
    #     all_tokens.extend(gpt2_pre_tokenize(sub_chunk))
    return all_tokens

def _update_stats(
    word: bytes,
    split: list[bytes],
    word_freq: int,
    pair_freqs: defaultdict[tuple[bytes, bytes], int],
    pair_to_words: defaultdict[tuple[bytes, bytes], set[bytes]],
    is_add: bool
):
    """
    一个辅助函数，用于从频率和索引中添加或删除一个词的所有对。
    is_add=True  -> 添加
    is_add=False -> 移除
    """
    for i in range(len(split) - 1):
        pair = (split[i], split[i + 1])
        delta = word_freq if is_add else -word_freq
        
        pair_freqs[pair] += delta
        if pair_freqs[pair] <= 0:
            # 安全地删除，以防万一
            if pair in pair_freqs:
                del pair_freqs[pair]
        
        # 更新倒排索引，初始时 words_set 等于空集合 set()
        words_set = pair_to_words[pair]
        if is_add:
            # pair_to_words[pair] 的 value 也会变成 word 不再是 set()
            words_set.add(word)
        elif word in words_set:
            words_set.remove(word)
        
        if not words_set:
            if pair in pair_to_words:
                del pair_to_words[pair]

# 将 count_tokens_in_chunk 移到模块级别
def count_tokens_in_chunk(start: int, end: int, input_path: str, special_tokens: list[str]) -> dict[bytes, int]:
    """统计文件块中的token频率
    返回一个字典，键为token的字节序列，值为该token在块中的出现频率。
    """
    # 初始化默认字典用于存储频率统计
    freq: dict[bytes, int] = defaultdict(int)
    # 将特殊token转换为字节序列
    special_token_bytes = {tok.encode("utf-8") for tok in special_tokens}

    # 打开输入文件
    with open(input_path, 'rb') as f:
        # 使用内存映射提高大文件读取效率
        mmapped = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        # 移动到块的起始位置
        mmapped.seek(start)
        # 读取块内容并解码为Unicode字符串
        chunk = mmapped.read(end - start).decode('utf-8', errors='ignore')
        # 对块进行分词
        tokens = tokenize_chunk((chunk, special_tokens))  # 返回list[str]
        # 统计每个token的频率
        for tok in tokens:
            # 将token转换为字节序列
            b = tok.encode('utf-8')
            # 跳过特殊token
            if b in special_token_bytes:
                continue
            # 增加token频率计数
            freq[b] += 1
    return freq

def merge_token_sequence(token_seq: Tuple, best_pair: Tuple, new_token: bytes) -> Tuple:
    """在一个token序列中，将所有出现的 best_pair 合并为 new_token"""
    new_seq = []
    i = 0
    while i < len(token_seq):
        # 检查当前位置是否是最佳对的开始
        if i < len(token_seq) - 1 and (token_seq[i], token_seq[i+1]) == best_pair:
            new_seq.append(new_token)
            i += 2
        else:
            new_seq.append(token_seq[i])
            i += 1
    return tuple(new_seq)




def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """训练BPE分词器，并返回词汇表和合并规则。
    参数:
        input_path: BPE分词器训练数据的文件路径
        vocab_size: 词汇表总大小（包含特殊token）
        special_tokens: 特殊token列表，这些token不会被分割
            这些token将始终作为单个token保留。如果这些特殊token出现在input_path中，
            它们会被当作普通字符串处理。

    返回:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab: 训练好的分词器词汇表，从token ID映射到token字节序列
            merges: BPE合并规则列表。每个列表项是一个字节元组(token1, token2)，
                   表示token1和token2被合并。合并规则按创建顺序排列。
            装在一个不可改动的Tuple中返回。
    """
    # 设置并行处理参数
    num_processes = 8
    num_chunks = 4 * num_processes
    # 将特殊token转换为字节序列
    special_token_bytes = {tok.encode("utf-8") for tok in special_tokens}
    
    # 读取文件并找到分块边界
    with open(input_path, 'rb') as f:
        boundaries = find_chunk_boundaries(f, num_chunks, special_tokens[0].encode('utf-8'))
    
    # 并行处理：统计每个块中token的频率
    results = Parallel(n_jobs=num_processes)(
        delayed(count_tokens_in_chunk)(start, end, input_path, special_tokens)
        for start, end in zip(boundaries[:-1], boundaries[1:])
    )
    
    # 初始化数据结构
    vocab: dict[int, bytes] = {}      # 词汇表：token_id -> token_bytes
    merges: list[tuple[bytes, bytes]] = []  # 合并规则列表
    token_freqs = defaultdict(int)    # token频率统计
    
    # 合并所有块的统计结果
    for local_freq in results:
        for token, count in local_freq.items():
            token_freqs[token] += count
    
    # 首先将特殊token添加到词汇表
    token_id = 0
    for special_token in special_tokens:
        vocab[token_id] = special_token.encode('utf-8')
        token_id += 1
    
    # 添加基础字节到词汇表（0-255）
    for i in range(256):
        b = bytes([i])
        if b not in special_token_bytes:
            vocab[token_id] = b
            token_id += 1
    
    # 初始化词的分割状态：每个词初始被分割成单个字节
    split_freqs: dict[bytes, list[bytes]] = {token: [bytes([b]) for b in token] for token in token_freqs}
    # 用于存储相邻对的频率
    pair_freqs: defaultdict[tuple[bytes, bytes], int] = defaultdict(int)
    # 用于存储每个相邻对出现在哪些词中
    pair_to_words: defaultdict[tuple[bytes, bytes], set[bytes]] = defaultdict(set)

    # 初始化统计信息
    for word, split in split_freqs.items():
        word_freq = token_freqs[word]
        _update_stats(word, split, word_freq, pair_freqs, pair_to_words, is_add=True)

    # 计算需要执行的合并次数
    nvocab = len(vocab)
    num_merges = vocab_size - nvocab
    
    # 主循环：执行BPE合并
    for i in tqdm(range(num_merges)):
        if not pair_freqs:  # 如果没有可合并的对了就退出
            break
        # 选择频率最高的对进行合并
        best_pair = max(pair_freqs.items(), key=lambda x: (x[1], x[0]))[0]
 
        # 记录合并规则并创建新token
        merges.append(best_pair)
        new_token = best_pair[0] + best_pair[1]
        vocab[nvocab] = new_token
        nvocab += 1
        
        # 获取需要被合并的token对
        p1, p2 = best_pair
        # 获取所有包含这对token的词
        words_to_update = list(pair_to_words[best_pair])

        # 更新受影响的词的分割状态
        for word in words_to_update:
            word_freq = token_freqs[word]
            old_split = split_freqs[word]
            
            # 移除旧的统计信息
            _update_stats(word, old_split, word_freq, pair_freqs, pair_to_words, is_add=False)
            
            # 执行合并操作
            new_split = []
            i = 0
            while i < len(old_split):
                if i < len(old_split) - 1 and old_split[i] == p1 and old_split[i+1] == p2:
                    new_split.append(new_token)
                    i += 2
                else:
                    new_split.append(old_split[i])
                    i += 1
            
            # 更新分割状态和统计信息
            split_freqs[word] = new_split
            _update_stats(word, new_split, word_freq, pair_freqs, pair_to_words, is_add=True)
    
    return vocab, merges



if  __name__ == "__main__":

    input_path = '/home/aeon/Desktop/Learn/CS336/Assignment/assignment1-basics/data/TinyStoriesV2-GPT4-train.txt'
    vocab_size = 10000
    special_tokens = ['<|endoftext|>']
    print("开始训练BPE分词器...")
    print(f"输入文件: {input_path}")
    print(f"目标词汇表大小: {vocab_size}")
    print(f"特殊token: {special_tokens}")
    print("-" * 50)


    vocab, merges = train_bpe(
        input_path=input_path,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
    )
    
    print(f"训练完成！实际词汇表大小: {len(vocab)}")
    print(f"合并操作次数: {len(merges)}")
    print("-" * 50)

    # 打印词汇表的一部分（前20个和后20个）
    print("词汇表示例（前20个）：")
    vocab_items = list(vocab.items())
    for i, (token_id, token_bytes) in enumerate(vocab_items[:20]):
        try:
            decoded = token_bytes.decode('utf-8')
            print(f"ID {token_id:3d}: '{decoded}' ({repr(token_bytes)})")
        except UnicodeDecodeError:
            print(f"ID {token_id:3d}: {repr(token_bytes)} (无法解码)")

    if len(vocab) > 40:
        print("\n...")
        print(f"(省略 {len(vocab) - 40} 个词汇)")
        print("...\n")
        
        print("词汇表示例（最后20个）：")
        for token_id, token_bytes in vocab_items[-20:]:
            try:
                decoded = token_bytes.decode('utf-8')
                print(f"ID {token_id:3d}: '{decoded}' ({repr(token_bytes)})")
            except UnicodeDecodeError:
                print(f"ID {token_id:3d}: {repr(token_bytes)} (无法解码)")

    # 打印合并操作的一部分（前10个和后10个）
    print("\n" + "-" * 50)
    print("合并操作示例（前10个）：")
    for i, (a, b) in enumerate(merges[:10]):
        try:
            a_decoded = a.decode('utf-8', errors='replace')
            b_decoded = b.decode('utf-8', errors='replace')
            merged = (a + b).decode('utf-8', errors='replace')
            print(f"合并 {i+1:2d}: '{a_decoded}' + '{b_decoded}' → '{merged}'")
        except Exception as e:
            print(f"合并 {i+1:2d}: {repr(a)} + {repr(b)} (解码错误: {e})")

    if len(merges) > 20:
        print("\n...")
        print(f"(省略 {len(merges) - 20} 个合并操作)")
        print("...\n")
        
        print("合并操作示例（最后10个）：")
        for i, (a, b) in enumerate(merges[-10:], len(merges) - 9):
            try:
                a_decoded = a.decode('utf-8', errors='replace')
                b_decoded = b.decode('utf-8', errors='replace')
                merged = (a + b).decode('utf-8', errors='replace')
                print(f"合并 {i:2d}: '{a_decoded}' + '{b_decoded}' → '{merged}'")
            except Exception as e:
                print(f"合并 {i:2d}: {repr(a)} + {repr(b)} (解码错误: {e})")

    # 定义保存路径
    TOKENIZER_DIR = '/home/aeon/Desktop/Learn/CS336/Assignment/assignment1-basics/tokenizer_models'
    VOCAB_PATH = os.path.join(TOKENIZER_DIR, 'vocab.pkl')
    MERGES_PATH = os.path.join(TOKENIZER_DIR, 'merges.pkl')
    
    # 序列化到磁盘
    print("\n" + "-" * 50)
    print("保存模型到本地...")
    os.makedirs(TOKENIZER_DIR, exist_ok=True)
    
    with open(VOCAB_PATH, "wb") as f:
        pickle.dump(vocab, f)
    print(f"词汇表已保存到: {VOCAB_PATH}")
    
    with open(MERGES_PATH, "wb") as f:
        pickle.dump(merges, f)
    print(f"合并操作已保存到: {MERGES_PATH}")

    # 同时保存人类可读的版本
    VOCAB_TXT_PATH = os.path.join(TOKENIZER_DIR, 'vocab.txt')
    MERGES_TXT_PATH = os.path.join(TOKENIZER_DIR, 'merges.txt')
    
    # 保存词汇表为文本文件
    with open(VOCAB_TXT_PATH, "w", encoding="utf-8") as f:
        f.write("# BPE词汇表\n")
        f.write(f"# 词汇表大小: {len(vocab)}\n")
        f.write("# 格式: token_id\\ttoken_bytes\\tdecoded_string\n\n")
        for token_id, token_bytes in sorted(vocab.items()):
            try:
                decoded = token_bytes.decode('utf-8')
                f.write(f"{token_id}\t{repr(token_bytes)}\t{repr(decoded)}\n")
            except UnicodeDecodeError:
                f.write(f"{token_id}\t{repr(token_bytes)}\t<无法解码>\n")
    
    # 保存合并操作为文本文件
    with open(MERGES_TXT_PATH, "w", encoding="utf-8") as f:
        f.write("# BPE合并操作\n")
        f.write(f"# 合并次数: {len(merges)}\n")
        f.write("# 格式: merge_index\\ttoken1\\ttoken2\\tmerged_result\n\n")
        for i, (a, b) in enumerate(merges):
            try:
                a_decoded = a.decode('utf-8', errors='replace')
                b_decoded = b.decode('utf-8', errors='replace')
                merged = (a + b).decode('utf-8', errors='replace')
                f.write(f"{i}\t{repr(a_decoded)}\t{repr(b_decoded)}\t{repr(merged)}\n")
            except Exception as e:
                f.write(f"{i}\t{repr(a)}\t{repr(b)}\t<解码错误: {e}>\n")
    
    print(f"词汇表文本版本已保存到: {VOCAB_TXT_PATH}")
    print(f"合并操作文本版本已保存到: {MERGES_TXT_PATH}")
    # 保存训练配置
    CONFIG_PATH = os.path.join(TOKENIZER_DIR, 'config.txt')
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        f.write("# BPE分词器训练配置\n")
        f.write(f"输入文件: {input_path}\n")
        f.write(f"词汇表大小: {vocab_size}\n")
        f.write(f"实际词汇表大小: {len(vocab)}\n")
        f.write(f"特殊token: {special_tokens}\n")
        f.write(f"合并操作次数: {len(merges)}\n")
        f.write(f"训练完成时间: {__import__('datetime').datetime.now()}\n")
    
    print(f"训练配置已保存到: {CONFIG_PATH}")
    print("\n训练完成！所有文件已保存到:", TOKENIZER_DIR)

    # 验证保存的文件
    print("\n" + "-" * 50)
    print("验证保存的文件:")
    for filename in ['vocab.pkl', 'merges.pkl', 'vocab.txt', 'merges.txt', 'config.txt']:
        filepath = os.path.join(TOKENIZER_DIR, filename)
        if os.path.exists(filepath):
            size = os.path.getsize(filepath)
            print(f"✓ {filename}: {size:,} 字节")
        else:
            print(f"✗ {filename}: 文件不存在")
    # 统计最长 token
    longest_token = max(vocab.values(), key=len)
    print("最长token:", longest_token, "长度:", len(longest_token))