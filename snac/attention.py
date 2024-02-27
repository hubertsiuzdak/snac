import torch
from einops import rearrange
from torch import nn


class LocalMHA(nn.Module):
    def __init__(self, dim=1024, window_size=32, dim_head=64, use_rotary_pos_emb=True):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.heads = dim // dim_head
        self.window_size = window_size
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        if use_rotary_pos_emb:
            self.rel_pos = SinusoidalEmbeddings(dim_head, scale_base=window_size // 2)
        else:
            self.rel_pos = None
        self.to_out = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        B, C, T = x.shape
        residual = x
        x = self.norm(x.transpose(1, 2))
        windows = T // self.window_size
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b (w n) (h d) -> b h w n d", w=windows, h=self.heads), (q, k, v))
        if self.rel_pos is not None:
            pos_emb, scale = self.rel_pos(k)
            q, k = apply_rotary_pos_emb(q, k, pos_emb, scale)
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        out = rearrange(out, "b h w n d -> b (w n) (h d)")
        out = self.to_out(out)
        return out.transpose(1, 2) + residual


class SinusoidalEmbeddings(nn.Module):
    def __init__(self, dim, scale_base=None, use_xpos=False):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        # xpos related
        self.use_xpos = use_xpos
        self.scale_base = scale_base
        assert not (use_xpos and scale_base is None), "scale base must be defined if using xpos"
        scale = (torch.arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)
        self.register_buffer("scale", scale, persistent=False)

    def forward(self, x):
        seq_len, device = x.shape[-2], x.device
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum("i , j -> i j", t, self.inv_freq)
        freqs = torch.cat((freqs, freqs), dim=-1)
        if not self.use_xpos:
            return freqs, torch.ones(1, device=device)
        power = (t - (seq_len // 2)) / self.scale_base
        scale = self.scale ** rearrange(power, "n -> n 1")
        scale = torch.cat((scale, scale), dim=-1)

        return freqs, scale


def rotate_half(x):
    x = rearrange(x, "b ... (r d) -> b ... r d", r=2)
    x1, x2 = x.unbind(dim=-2)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, freqs, scale=1):
    q_len = q.shape[-2]
    q_freqs = freqs[..., -q_len:, :]
    inv_scale = scale**-1
    if scale.ndim == 2:
        scale = scale[-q_len:, :]
    q = (q * q_freqs.cos() * scale) + (rotate_half(q) * q_freqs.sin() * scale)
    k = (k * freqs.cos() * inv_scale) + (rotate_half(k) * freqs.sin() * inv_scale)
    return q, k
