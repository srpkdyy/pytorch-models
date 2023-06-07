import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from einops import rearrange
from einops.layers.torch import Rearrange


class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_pos=512):
        super().__init__()

        pos = torch.arange(max_pos)

        freq = torch.arange(dim//2) / dim
        freq = (freq * torch.tensor(10000).log()).exp()

        x = rearrange(pos, 'L -> L 1') / freq
        x = rearrange(x, 'L d -> L d 1')

        pe = torch.cat((x.sin(), x.cos()), dim=-1)
        self.pe = rearrange(pe, 'L d sc -> L (d sc)')

    def forward(self, n, *, device=torch.device('cpu')):
        enc = self.pe[:n]
        return enc.to(device)


class ScaledDotProductAttention(nn.Module):
    def __init__(self, dim, p_drop):
        super().__init__()

        self.scale = dim ** 0.5
        self.fill_val = torch.tensor(-float('inf'))

        self.dropout = nn.Dropout(p_drop)

    def forward(self, q, k, v, mask=None):
        qk = torch.einsum('...id,...jd->...ij', q, k)
        scaled_qk = qk / self.scale

        if mask is not None:
            scaled_qk.masked_fill_(mask, self.fill_val)

        attn = F.softmax(scaled_qk, dim=-1)
        attn = self.dropout(attn)

        out = einsum('...ij,...jd->...id', attn, v)
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, n_heads, p_drop):
        super().__init__()

        self.n_heads = n_heads

        kv_dim = dim // n_heads
        proj_dim = kv_dim * n_heads

        self.q_proj = nn.Linear(dim, proj_dim, bias=False)
        self.k_proj = nn.Linear(dim, proj_dim, bias=False)
        self.v_proj = nn.Linear(dim, proj_dim, bias=False)

        self.attention = ScaledDotProductAttention(kv_dim, p_drop)

        self.out_proj = nn.Linear(proj_dim, dim, bias=False)
        self.dropout = nn.Dropout(p_drop)

    def forward(self, q, kv=None, mask=None):
        kv = kv if kv is not None else q

        qkv = torch.stack([self.q_proj(q), self.k_proj(kv), self.v_proj(kv)])
        q, k, v = rearrange(qkv, 'qkv b l (h d) -> qkv b h l d', h=self.n_heads)

        attn = self.attention(q, k, v, mask=mask)
        attn = rearrange(attn, 'b h l d -> b l (h d)')

        out = self.out_proj(attn)
        out = self.dropout(out)
        return out


class FeedForwardNet(nn.Module):
    def __init__(self, dim, p_drop):
        super().__init__()

        ff_dim = dim * 4

        self.net = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, dim),
            nn.Dropout(p_drop)
        )

    def forward(self, x):
        return self.net(x)


class EncoderLayer(nn.Module):
    def __init__(self, dim, n_heads, p_drop):
        super().__init__()

        self.mha = MultiHeadAttention(dim, n_heads, p_drop)
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = FeedForwardNet(dim, p_drop)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.norm1(x + self.mha(x))
        x = self.norm2(x + self.ffn(x))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, dim, n_heads, p_drop):
        super().__init__()

        self.mask_mha = MultiHeadAttention(dim, n_heads, p_drop)
        self.norm1 = nn.LayerNorm(dim)
        self.mha = MultiHeadAttention(dim, n_heads, p_drop)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = FeedForwardNet(dim, p_drop)
        self.norm3 = nn.LayerNorm(dim)

    def forward(self, x, kv=None):
        B, L, D = x.shape

        mask = torch.ones(L, L, dtype=torch.bool, device=x.device)
        mask = mask.tril().logical_not()

        x = self.norm1(x + self.mask_mha(x, mask=mask))
        x = self.norm2(x + self.mha(x, kv=kv))
        x = self.norm3(x + self.ffn(x))
        return x


class Transformer(nn.Module):
    def __init__(self,
                 n_layers=6,
                 dim=512,
                 ff_dim=2048,
                 n_heads=8,
                 kv_dim=64,
                 p_drop=0.1,
                 ):
        super().__init__()

        self.pos_emb = PositionalEncoding(dim)

        self.enc_drop = nn.Dropout(p_drop)
        self.dec_drop = nn.Dropout(p_drop)

        self.encoder = nn.ModuleList([
            EncoderLayer(dim, n_heads, p_drop) for _ in range(n_layers)
        ])
        self.decoder = nn.ModuleList([
            DecoderLayer(dim, n_heads, p_drop) for _ in range(n_layers)
        ])

    def forward(self, enc_x, dec_x):
        enc_x = enc_x + self.pos_emb(enc_x.shape[1], device=enc_x.device)
        enc_x = self.enc_drop(enc_x)

        for layer in self.encoder:
            enc_x = layer(enc_x)

        dec_x = dec_x + self.pos_emb(dec_x.shape[1], device=dec_x.device)
        dec_x = self.dec_drop(dec_x)

        for layer in self.decoder:
            dec_x = layer(dec_x, kv=dec_x)

        return dec_x


if __name__ == '__main__':
    model = Transformer()
    out = model(torch.rand(2, 3, 512), torch.rand(2, 3, 512))
    print(out.shape)

