"""Encoder-decoder Transformer with a modern block: pre-RMSNorm, RoPE, SwiGLU, GQA.

One shared vocabulary for en and uz; the target language is chosen by a
<2xx> tag at the start of the source sentence.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class ModelConfig:
    vocab_size: int = 16384
    d_model: int = 640
    n_heads: int = 10
    n_kv_heads: int = 10
    enc_layers: int = 8  # deep encoder, shallow decoder
    dec_layers: int = 4
    ffn_dim: int = 1728
    dropout: float = 0.1
    max_len: int = 256
    pad_id: int = 0
    rope_base: float = 10000.0
    tie_embeddings: bool = True


class RoPE(nn.Module):
    def __init__(self, head_dim: int, max_len: int, base: float):
        super().__init__()
        inv = 1.0 / base ** (torch.arange(0, head_dim, 2).float() / head_dim)
        freqs = torch.outer(torch.arange(max_len).float(), inv)
        self.register_buffer("cos", freqs.cos(), persistent=False)
        self.register_buffer("sin", freqs.sin(), persistent=False)

    def forward(self, x: torch.Tensor, start: int = 0) -> torch.Tensor:
        t = x.size(2)
        cos = self.cos[start : start + t].to(x.dtype)
        sin = self.sin[start : start + t].to(x.dtype)
        x1, x2 = x[..., ::2], x[..., 1::2]
        return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2)


def _heads(x: torch.Tensor, n: int) -> torch.Tensor:
    b, t, _ = x.shape
    return x.view(b, t, n, -1).transpose(1, 2)


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.h, self.kvh = cfg.n_heads, cfg.n_kv_heads
        hd = cfg.d_model // cfg.n_heads
        self.wq = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.wkv = nn.Linear(cfg.d_model, 2 * self.kvh * hd, bias=False)
        self.wo = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.dropout = cfg.dropout

    def kv(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        k, v = self.wkv(x).chunk(2, dim=-1)
        return _heads(k, self.kvh), _heads(v, self.kvh)

    def attend(self, q, k, v, mask=None, is_causal=False) -> torch.Tensor:
        if self.kvh != self.h:
            k = k.repeat_interleave(self.h // self.kvh, dim=1)
            v = v.repeat_interleave(self.h // self.kvh, dim=1)
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, is_causal=is_causal,
            dropout_p=self.dropout if self.training else 0.0,
        )
        b, _, t, _ = out.shape
        return self.wo(out.transpose(1, 2).reshape(b, t, -1))


class SelfAttention(Attention):
    def forward(self, x, rope: RoPE, mask=None, is_causal=False, cache=None, start_pos=0):
        q = rope(_heads(self.wq(x), self.h), start_pos)
        k, v = self.kv(x)
        k = rope(k, start_pos)
        if cache is not None:
            if "k" in cache:
                k = torch.cat((cache["k"], k), dim=2)
                v = torch.cat((cache["v"], v), dim=2)
            cache["k"], cache["v"] = k, v
        return self.attend(q, k, v, mask, is_causal)


class CrossAttention(Attention):
    def forward(self, x, memory, mask, cache=None):
        q = _heads(self.wq(x), self.h)
        if cache is not None and "ck" in cache:
            k, v = cache["ck"], cache["cv"]
        else:
            k, v = self.kv(memory)
            if cache is not None:
                cache["ck"], cache["cv"] = k, v
        return self.attend(q, k, v, mask)


class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.w1 = nn.Linear(cfg.d_model, cfg.ffn_dim, bias=False)
        self.w3 = nn.Linear(cfg.d_model, cfg.ffn_dim, bias=False)
        self.w2 = nn.Linear(cfg.ffn_dim, cfg.d_model, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.w2(self.drop(F.silu(self.w1(x)) * self.w3(x)))


class EncoderLayer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm1, self.norm2 = nn.RMSNorm(cfg.d_model), nn.RMSNorm(cfg.d_model)
        self.attn, self.ffn = SelfAttention(cfg), SwiGLU(cfg)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, rope, mask):
        x = x + self.drop(self.attn(self.norm1(x), rope, mask=mask))
        return x + self.drop(self.ffn(self.norm2(x)))


class DecoderLayer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm1, self.norm2, self.norm3 = (nn.RMSNorm(cfg.d_model) for _ in range(3))
        self.self_attn, self.cross_attn = SelfAttention(cfg), CrossAttention(cfg)
        self.ffn = SwiGLU(cfg)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, memory, mem_mask, rope, cache=None, start_pos=0):
        # Incremental decoding feeds one token at a time, so the causal mask is implicit.
        h = self.self_attn(self.norm1(x), rope, is_causal=cache is None,
                           cache=cache, start_pos=start_pos)
        x = x + self.drop(h)
        x = x + self.drop(self.cross_attn(self.norm2(x), memory, mem_mask, cache))
        return x + self.drop(self.ffn(self.norm3(x)))


class Seq2SeqTransformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_id)
        self.rope = RoPE(cfg.d_model // cfg.n_heads, cfg.max_len, cfg.rope_base)
        self.encoder = nn.ModuleList(EncoderLayer(cfg) for _ in range(cfg.enc_layers))
        self.decoder = nn.ModuleList(DecoderLayer(cfg) for _ in range(cfg.dec_layers))
        self.enc_norm, self.dec_norm = nn.RMSNorm(cfg.d_model), nn.RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.drop = nn.Dropout(cfg.dropout)
        self.apply(self._init_weights)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed.weight

    @staticmethod
    def _init_weights(m: nn.Module):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def encode(self, src: torch.Tensor):
        mask = src.ne(self.cfg.pad_id)[:, None, None, :]
        x = self.drop(self.embed(src))
        for layer in self.encoder:
            x = layer(x, self.rope, mask)
        return self.enc_norm(x), mask

    def decode(self, tgt, memory, mem_mask, caches=None, start_pos=0):
        x = self.drop(self.embed(tgt))
        for i, layer in enumerate(self.decoder):
            cache = caches[i] if caches is not None else None
            x = layer(x, memory, mem_mask, self.rope, cache, start_pos)
        return self.lm_head(self.dec_norm(x))

    def forward(self, src: torch.Tensor, tgt_in: torch.Tensor) -> torch.Tensor:
        memory, mask = self.encode(src)
        return self.decode(tgt_in, memory, mask)

    def new_caches(self) -> list[dict]:
        return [{} for _ in self.decoder]

    @torch.no_grad()
    def generate(self, src, bos_id, eos_id, max_len=256, beam=4, alpha=0.6):
        """Beam search with KV cache and GNMT length penalty. beam=1 is greedy."""
        b, k = src.size(0), beam
        memory, mask = self.encode(src)
        memory, mask = memory.repeat_interleave(k, 0), mask.repeat_interleave(k, 0)
        dev = src.device

        tokens = torch.full((b * k, 1), bos_id, device=dev)
        scores = torch.full((b, k), float("-inf"), device=dev)
        scores[:, 0] = 0.0  # all beams start identical, keep only one alive
        scores = scores.view(-1)
        finished = torch.zeros(b * k, dtype=torch.bool, device=dev)
        lengths = torch.zeros(b * k, device=dev)
        caches = self.new_caches()
        offsets = (torch.arange(b, device=dev) * k)[:, None]

        max_len = min(max_len, self.cfg.max_len - 1)
        for step in range(max_len):
            logp = self.decode(tokens[:, -1:], memory, mask, caches, start_pos=step)
            logp = logp[:, -1].float().log_softmax(-1)
            vocab = logp.size(-1)
            done_row = torch.full_like(logp[0], float("-inf"))
            done_row[eos_id] = 0.0
            logp = torch.where(finished[:, None], done_row, logp)

            top, idx = (scores[:, None] + logp).view(b, k * vocab).topk(k, dim=-1)
            sel = (offsets + idx // vocab).view(-1)
            tok = (idx % vocab).view(-1)

            prev_done = finished[sel]
            tokens = torch.cat((tokens[sel], tok[:, None]), dim=1)
            lengths = lengths[sel] + (~prev_done).float()
            finished = prev_done | tok.eq(eos_id)
            scores = top.view(-1)
            for c in caches:
                for name in c:
                    c[name] = c[name].index_select(0, sel)
            if finished.all():
                break

        penalty = ((5.0 + lengths) / 6.0) ** alpha
        best = (scores / penalty).view(b, k).argmax(-1)
        return tokens.view(b, k, -1)[torch.arange(b, device=dev), best, 1:]
