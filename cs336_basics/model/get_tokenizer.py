## 实现作业中的函数，实际上是要写一个tokenizer类
import pickle
from typing import Any, Iterable, Iterator
import regex as re  # 使用regex库来处理Unicode字符集
from tqdm import tqdm  # 用于进度条显示
import collections
from collections import defaultdict

class Tokenizer():
    ''''
    BPE分词器类，使用给定的词汇表和合并规则进行文本编码和解码。
    '''
    def __init__(self,vocab,merges,special_tokens=None):
        # 直接接收 dict[int, bytes] 和 list[tuple[bytes, bytes]]
        self.vocab = vocab.copy()
        self.merges_rank = {merge: i for i, merge in enumerate(merges)}
        self.special_token_bytes = set()
        if special_tokens:
            self.special_token_bytes = {st.encode('utf-8') for st in special_tokens}
            self.add_special_tokens(special_tokens)
        self.vocab_to_id  = {v: k for k, v in self.vocab.items()}

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None): 
    # 加载 vocab.pkl
        with open(vocab_filepath, 'rb') as vf:
                    raw_vocab = pickle.load(vf)
        # 转换为 {int: bytes}
        vocab = {int(k): (v.encode("utf-8") if isinstance(v, str) else v)
                for k, v in raw_vocab.items()}
        
        # 加载 merges.pkl
        with open(merges_filepath, 'rb') as mf:
            raw_merges = pickle.load(mf)
        # 转换为 List[Tuple[bytes, bytes]]
        merges = []
        for a, b in raw_merges:
            merges.append((
                a.encode("utf-8") if isinstance(a, str) else a,
                b.encode("utf-8") if isinstance(b, str) else b
            ))
        return cls(vocab, merges, special_tokens)


    def encode(self, text: str) -> list[int]: 
        '''
        编码文本为token ID序列

        '''
        tokens: list[bytes] = []
        if self.special_token_bytes:
            # ← 关键：先按 bytes 长度倒序，保证重叠时最长的先匹配
            specials = sorted(self.special_token_bytes, key=lambda b: -len(b))
            # 解码到 str，再 escape 拼正则
            pattern = "|".join(re.escape(st.decode("utf-8")) for st in specials)
            chunks = re.split(f"({pattern})", text)
        else:
            chunks = [text]

        for chunk in chunks:
            if not chunk:
                continue
            chunk_b = chunk.encode("utf-8")
            if chunk_b in self.special_token_bytes:
                tokens.append(chunk_b)
            else:
                for pre in gpt2_pre_tokenize(chunk):
                    bpre = pre.encode("utf-8")
                    for tk in self.merge_bytes(bpre):
                        tokens.append(tk)

        # 最后把 bytes token 转成 ID 列表
        return [self.vocab_to_id[t] for t in tokens]

    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        for it in tqdm(iterable):
            yield from self.encode(it)

    def decode(self, ids: list[int]) -> str:
        ret = b"".join(self.vocab[it] for it in ids).decode('utf-8', errors="replace")
        return ret

    def add_special_tokens(self, special_tokens: list[str]):
        exit_vocab = set(self.vocab.values())
        new_special_tokens = [st.encode('utf-8') for st in special_tokens if st.encode('utf-8') not in exit_vocab]
        if new_special_tokens:
            max_key = max(self.vocab.keys())
            self.vocab.update({max_key + i + 1: nst for i, nst in enumerate(new_special_tokens)})

    def merge_bytes(self, token: bytes) -> list[bytes]:
        split = [bytes([tk]) for tk in token]
        while True:
            pair_rank: defaultdict[tuple[bytes, bytes], int] = defaultdict(int)
            for i in range(len(split) - 1):
                pair_rank[(split[i], split[i + 1])] = self.merges_rank.get((split[i], split[i + 1]), float("inf"))
            if not pair_rank:
                break
            best_pair = min(pair_rank.items(), key=lambda x: (x[1], -int.from_bytes(x[0][0] + x[0][1], "big")))[0]
            if best_pair not in self.merges_rank:
                break
            new_split = []
            i = 0
            while i < len(split):
                if i < len(split) - 1 and split[i] == best_pair[0] and split[i + 1] == best_pair[1]:
                    new_split.append(split[i] + split[i + 1])
                    i += 2
                else:
                    new_split.append(split[i])
                    i += 1
            split = new_split
        return split

def gpt2_pre_tokenize(text: str) -> list:
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    return [m.group() for m in re.finditer(PAT, text)]


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
    
    tokenizer = Tokenizer(vocab, merges, special_tokens)
    return tokenizer

if __name__ == "__main__":
    # Example usage
    vocab = "/home/aeon/Desktop/Learn/CS336/Assignment/assignment1-basics/tokenizer_models/vocab.pkl"
    merges = "/home/aeon/Desktop/Learn/CS336/Assignment/assignment1-basics/tokenizer_models/merges.pkl"

    tokenizer = Tokenizer.from_files(vocab, merges, special_tokens=["<|endoftext|>"])

    text = "Hello, world! This is a test."
    encoded = tokenizer.encode(text)
    print("Encoded:", encoded)
    decoded = tokenizer.decode(encoded)
    print("Decoded:", decoded)
    # 测试编码和解码