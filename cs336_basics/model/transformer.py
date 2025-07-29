import os
import json
from pathlib import Path
import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float, Int
from cs336_basics.model.embedding import Embedding
from cs336_basics.model.msa import run_softmax
from cs336_basics.model.rope import RotaryPositionalEmbedding
from cs336_basics.model.linear import Linear
from cs336_basics.model.rmsnorm import RMSNorm
from cs336_basics.model.swiglu import SwiGLU
from cs336_basics.model.msa import Multihead_self_attention

class Transformer_block(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float | None = None):
        super().__init__()
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.theta = theta
        if theta is not None:
            pos_encode = RotaryPositionalEmbedding(theta, d_model // num_heads, max_seq_len)
            self.attn = Multihead_self_attention(d_model=d_model, num_heads=num_heads, pos_encode=pos_encode, theta=theta)
        else:
            self.attn = Multihead_self_attention(d_model=d_model, num_heads=num_heads)
        self.rmsn_1 = RMSNorm(d_model=d_model, eps=1e-5)
        self.rmsn_2 = RMSNorm(d_model=d_model, eps=1e-5)
        # self.pw_ffn = SiLU(d_model=d_model, d_ff=d_ff)
        self.pw_ffn = SwiGLU(d_model=d_model, d_ff=d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn = self.attn(self.rmsn_1(x))
        out1 = x + attn
        out2 = self.pw_ffn(self.rmsn_2(out1))
        out = out1 + out2
        return out

class Transformer_lm(nn.Module):
    def __init__(self, vocab_size:int, context_length:int, num_layers: int, d_model: int, num_heads: int, d_ff: int, rope_theta: float | None = None):
        super().__init__()
        self.context_length = context_length
        self.transformer = nn.ModuleDict(dict(
            token_emb = Embedding(num_embedding=vocab_size, embedding_dim=d_model),
            n_block = nn.ModuleList([Transformer_block(d_model=d_model, num_heads=num_heads, d_ff=d_ff, max_seq_len=context_length, theta=rope_theta) for _ in range(num_layers)]),
            rmsn_l = RMSNorm(d_model=d_model, eps=1e-5)
        ))
        self.linear_emb = Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tkemb = self.transformer.token_emb(x)
        for block in self.transformer.n_block:
            tkemb = block(tkemb)
        tkemb = self.transformer.rmsn_l(tkemb)
        out = self.linear_emb(tkemb)
        return out

    @torch.no_grad()
    def generate(self, x: torch.Tensor, max_gen_tokens: int, temperature: float = 1.0, top_p: int | None = None, eos_token_id: int | None = None):
        if x.dim() == 1:
            x = x.unsqueeze(0)
            
        original_sequence_length = x.size(-1)
        for _ in range(max_gen_tokens):
            x = x[:, -self.context_length :] if x.size(1) > self.context_length else x
            logits = self.forward(x)
            next_token_logits = logits[:, -1, :]
            temperature_scaled = next_token_logits / temperature
            if top_p:
                sorted_logits, sorted_indices = torch.sort(temperature_scaled, descending=True)
                sorted_probs = run_softmax(sorted_logits, dim=-1)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                sorted_mask = cumulative_probs > top_p
                sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
                sorted_mask[..., 0] = False
                mask = sorted_mask.scatter(1, sorted_indices, sorted_mask)
                temperature_scaled = temperature_scaled.masked_fill(mask, float("-inf"))
            probs = run_softmax(temperature_scaled, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1)
            if eos_token_id is not None and next_token_id.item() == eos_token_id:
                break
            x = torch.cat((x, next_token_id), dim=-1)
        return x[:, original_sequence_length:]

    @classmethod
    def from_pretrained(cls, pretrained_path: str):
        with open(os.path.join(pretrained_path, "model_config.json")) as f:
            config = json.load(f)
        model = cls(**config)
        weights_path = os.path.join(pretrained_path, "model.pt")
        state_dict = torch.load(weights_path, weights_only=True)
        # Remove _orig_mod. prefix that comes from serializing a compiled model
        unwanted_prefix = "_orig_mod."
        for k, _ in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        return model

    def save_pretrained(self, pretrained_path: str):
        os.makedirs(pretrained_path, exist_ok=True)
        config = {
            "vocab_size": self.transformer["token_emb"].weight.size(0),
            "context_length": self.context_length,
            "num_layers": len(self.transformer["n_block"]),
            "d_model": self.transformer["token_emb"].weight.size(1),
            "num_heads": self.transformer["n_block"][0].num_heads,
            "d_ff": self.transformer["n_block"][0].pw_ffn.d_ff,
            "rope_theta": self.transformer["n_block"][0].theta
        }
        with open(Path(pretrained_path) / "model_config.json", "w") as f:
            json.dump(config, f, indent=4)
        torch.save(self.state_dict(), Path(pretrained_path) / "model.pt")
