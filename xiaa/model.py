import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .config import XiaaConfig


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        norm = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * norm * self.weight


def build_rope(seq_len: int, head_dim: int, theta: float, device):
    pos = torch.arange(seq_len, device=device)
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device) / head_dim))
    emb = torch.outer(pos, freqs)
    return torch.stack([emb.cos(), emb.sin()], dim=-1)  # (T, head_dim/2, 2)


def apply_rope(x, rope):
    # x: (B, H, T, D)  rope: (T, D/2, 2)
    d = x.shape[-1]
    x1 = x[..., :d//2]
    x2 = x[..., d//2:]
    cos = rope[..., 0].unsqueeze(0).unsqueeze(0)
    sin = rope[..., 1].unsqueeze(0).unsqueeze(0)
    return torch.cat([x1*cos - x2*sin, x1*sin + x2*cos], dim=-1)


class GQAttention(nn.Module):
    """Grouped Query Attention — same as LLaMA 2, Qwen"""
    def __init__(self, cfg: XiaaConfig):
        super().__init__()
        self.n_heads    = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim   = cfg.d_model // cfg.n_heads
        self.n_groups   = cfg.n_heads // cfg.n_kv_heads

        self.q  = nn.Linear(cfg.d_model, cfg.n_heads    * self.head_dim, bias=False)
        self.k  = nn.Linear(cfg.d_model, cfg.n_kv_heads * self.head_dim, bias=False)
        self.v  = nn.Linear(cfg.d_model, cfg.n_kv_heads * self.head_dim, bias=False)
        self.out = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x, rope):
        B, T, C = x.shape
        q = self.q(x).view(B, T, self.n_heads,    self.head_dim).transpose(1, 2)
        k = self.k(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q = apply_rope(q, rope)
        k = apply_rope(k, rope)

        # Expand KV heads to match Q heads
        k = k.repeat_interleave(self.n_groups, dim=1)
        v = v.repeat_interleave(self.n_groups, dim=1)

        # Flash attention (uses scaled dot product)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.out(y)


class SwiGLU(nn.Module):
    def __init__(self, cfg: XiaaConfig):
        super().__init__()
        h = int(cfg.d_model * cfg.ffn_mult)
        self.gate = nn.Linear(cfg.d_model, h, bias=False)
        self.up   = nn.Linear(cfg.d_model, h, bias=False)
        self.down = nn.Linear(h, cfg.d_model, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class TransformerBlock(nn.Module):
    def __init__(self, cfg: XiaaConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.attn  = GQAttention(cfg)
        self.norm2 = RMSNorm(cfg.d_model)
        self.ffn   = SwiGLU(cfg)

    def forward(self, x, rope):
        x = x + self.attn(self.norm1(x), rope)
        x = x + self.ffn(self.norm2(x))
        return x


class XiaaM1(nn.Module):
    def __init__(self, cfg: XiaaConfig):
        super().__init__()
        self.cfg = cfg
        self.embed   = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.layers  = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.norm    = RMSNorm(cfg.d_model)
        self.head    = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # Weight tying (saves ~120M params)
        self.head.weight = self.embed.weight

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        device = idx.device

        x    = self.embed(idx)
        rope = build_rope(T, self.cfg.d_model // self.cfg.n_heads,
                          self.cfg.rope_theta, device)

        for layer in self.layers:
            x = layer(x, rope)

        x    = self.norm(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.cfg.vocab_size),
                targets.view(-1),
                ignore_index=-1
            )
        return logits, loss

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=100, temperature=0.8, top_k=50):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.max_seq_len:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx