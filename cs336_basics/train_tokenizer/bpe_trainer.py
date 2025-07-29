import os
import regex
from typing import Tuple, List, Dict, Set
from collections import defaultdict 
import pickle


## better understand the code below, you need to know:
# txt文件本身是 UTF-8 编码的纯文本，你用 open(..., "rb") 读出来的是原始字节流。
# special_token 在代码里是 Python 字符串 "<|endoftext|>"


# “rb” 是 Python 打开文件时的模式，表示以 “二进制方式（read‐binary）” 读取文件，不做任何字符解码，直接返回原始的 bytes。
# 字节表示（bytes）就是一串原始的 0–255 的整数，每个整数对应一个字节。Python 用 bytes 类型来存这一段原始数据，通常写作以前缀 b 开头的字面量。
# 

# bytes 对象是 Python 中用来表示“原始二进制数据”的内置类型。
# 本质上，它是一个不可变的整数序列，每个元素的取值范围在 0–255（即一个字节）。
# bytes 不做字符编码/解码，存储的是原始的数值。原始二进制数据

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
    # Step 1: Initialize Vocabulary 初始化词表 
    # 字典 int: bytes对象。使用字典推导式，range(256)生成256个整数，bytes([i])表示把这个整数转换成bytes对象
    # 第1步初始化词汇表，基础词汇表包含所有256个基础字节，对应ASCII码范围是0-255
    # bytes([65]) 结果是 b'A'，即ASCII码65对应的字节 字典里：65: b'A', 
    vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    next_id = 256

    special_token_bytes = [token.encode("utf-8") for token in special_tokens]
    for token_bytes in special_token_bytes:
        if token_bytes not in vocab.values(): # 如果词表中没有这个特殊token
            vocab[next_id] = token_bytes #新建一个键值对 next_id加一
            next_id += 1

    # Step 2: Pre-tokenization 预分词

    # token_frequency_table: Dict[Tuple[bytes], int] = {} # 用于统计每个token出现的频率，注意不能用列表，只能用tuple元组，因为列表不可哈希
    # 会在一堆词里找，如果存在，频数加一；如果不存在，初始化为1
    # 这张表要得到 token ： 频数
    token_frequency_table = defaultdict(int) #总是存在没出现过的key，只好用defaultdict
    # 用一个集合来高效检查特殊符号的字节表示是否已存在于词汇表中，用列表也能查重，但时间复杂度是O(n)，集合是O(1)
    existing_byte_values: Set[bytes] = set(vocab.values())



    #以文本方式打开 训练数据文件，读取全部内容到text变量中
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            text = f.read()
    except FileNotFoundError:
        text = ""

    # 返回的 chunks 列表将包含标记之间的所有文本片段 相当于先进行一次段落分割
    chunks = regex.split("|".join(map(regex.escape, special_tokens)), text)
    # 结果：chunks = ["Hello world", "Another story", "End story"]
    
    # # 然后在大分割里小分割，按照空格和标点
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    for chunk in chunks:
        for match in regex.finditer(PAT, chunk):  # 获取match ”hello“
            word = match.group()  # 获取匹配的文本 “Hello”
            word_bytes = word.encode("utf-8")  # 将匹配到的单词转换为字节: b'Hello'
            bytes_list = [bytes([x]) for x in word_bytes] #e.g. ['h', 'e', 'l', 'l', 'o']
            token_frequency_table[tuple(bytes_list)] += 1   
    # token_frequency_table 的示例内容：
    # {
    #     (b'H', b'e', b'l', b'l', b'o'): 3,      # "Hello" 出现3次
    #     (b'w', b'o', b'r', b'l', b'd'): 2,      # "world" 出现2次
    #     (b't', b'h', b'e'): 5,                  # "the" 出现5次
    # }


    # Step 3: Compute BPE Merges
    merges: List[Tuple[bytes, bytes]] = [] # 用于存储合并操作记录
    
    #一次性统计所有token的对和频率
    pair_counts = defaultdict(int) # 用于存储字节对的频率

    for token in token_frequency_table.keys():
        for i in range(len(token) - 1):
            pair_counts[token[i], token[i+1]] += token_frequency_table[token]
#token[i], token[i+1]：获取相邻的两个字节，形成一个字节对
# token_frequency_table[token]：获取这个token的出现频率
# 将这个频率累加到对应字节对的计数中
# 最终的 pair_counts：
# {
#     (b'H', b'e'): 4,    # H-e 组合总共出现4次
#     (b'e', b'l'): 4,    # e-l 组合总共出现4次  
#     (b'l', b'l'): 3,    # l-l 组合总共出现3次
#     (b'l', b'o'): 3,    # l-o 组合总共出现3次
#     (b'l', b'p'): 1,    # l-p 组合总共出现1次
#     (b't', b'h'): 2,    # t-h 组合总共出现2次
#     (b'h', b'e'): 2,    # h-e 组合总共出现2次
# }


    # # 第4步开始训练BPE算法
    while len(vocab) < vocab_size: # 添加新的token直到词汇表达到指定大小
        if not pair_counts:
            break  # No more pairs to merge

        # Find the most frequent pair(s)
        max_count = max(pair_counts.values())
        candidates = [k for k, v in pair_counts.items() if v == max_count]
        # 在候选者中，选择字节序最大的那个
        best_pair = max(candidates)

        merges.append(best_pair)

        # Create new token
        new_token_bytes = best_pair[0] + best_pair[1] # 将最佳token对的两个token连接起来
        vocab[next_id] = new_token_bytes
        next_id += 1

        #记录受影响的token，也就是包含best_pair的来自token_frequency_table的token
        affected_tokens = []
        for token, freq in token_frequency_table.items():
            has_pair = any(token[i:i+2] == best_pair for i in range(len(token) - 1))
            if has_pair:
                affected_tokens.append((token, freq))
        #从受影响的token中出发,每个token就是token_frequency_table的key
        for token, freq in affected_tokens:
            # 删除pair_counts中对应的best_pair
            for i in range(len(token) - 1):
                pair_counts[token[i], token[i+1]] -= freq
                if pair_counts[token[i], token[i+1]] <= 0:
                    del pair_counts[token[i], token[i+1]]
            # 将best_pair合并为new_token
            new_token_frequency_seq = merge_token_sequence(token, best_pair, new_token_bytes)
            # 更新pair_counts
            for i in range(len(new_token_frequency_seq)-1):
                pair = (new_token_frequency_seq[i], new_token_frequency_seq[i+1])
                pair_counts[pair] += freq
            # 更新token_frequency_table
            del token_frequency_table[token]
            token_frequency_table[new_token_frequency_seq] += freq

    return vocab, merges # 返回最终的词汇表和合并记录




if  __name__ == "__main__":

    input_path = '/home/aeon/Desktop/Learn/CS336/Assignment/assignment1-basics/data/TinyStoriesV2-GPT4-train.txt'
    vocab_size = 10000
    special_tokens = ['<|endoftext|>']
    print("开始训练BPE分词器...")
    print(f"输入文件: {input_path}")
    print(f"目标词汇表大小: {vocab_size}")
    print(f"特殊token: {special_tokens}")
    print("-" * 50)


    vocab, merges = run_train_bpe(
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