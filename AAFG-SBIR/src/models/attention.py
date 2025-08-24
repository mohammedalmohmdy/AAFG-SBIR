import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention2d(nn.Module):
    """Lightweight 2D self-attention on feature maps (B,C,H,W)."""
    def __init__(self, channels, heads=4, reduction=1):
        super().__init__()
        inner = max(1, channels // max(1, reduction))
        self.q = nn.Conv2d(channels, inner, 1, bias=False)
        self.k = nn.Conv2d(channels, inner, 1, bias=False)
        self.v = nn.Conv2d(channels, channels, 1, bias=False)
        self.proj = nn.Conv2d(channels, channels, 1)
        self.ln = nn.LayerNorm([channels, 1, 1])

    def forward(self, x):
        B,C,H,W = x.shape
        q = self.q(x).flatten(2).transpose(1,2)   # B,HW,inner
        k = self.k(x).flatten(2).transpose(1,2)   # B,HW,inner
        v = self.v(x).flatten(2).transpose(1,2)   # B,HW,C
        attn = torch.matmul(q, k.transpose(1,2)) / (k.shape[-1] ** 0.5)  # B,HW,HW
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)  # B,HW,C
        out = out.transpose(1,2).reshape(B, C, H, W)
        out = self.proj(out)
        return self.ln(out + x)

class CrossAttention2d(nn.Module):
    """Bi-directional cross-attention between (sketch, image) feature maps."""
    def __init__(self, channels, reduction=1):
        super().__init__()
        inner = max(1, channels // max(1, reduction))
        self.qs = nn.Conv2d(channels, inner, 1, bias=False)
        self.ks = nn.Conv2d(channels, inner, 1, bias=False)
        self.vs = nn.Conv2d(channels, channels, 1, bias=False)
        self.qi = nn.Conv2d(channels, inner, 1, bias=False)
        self.ki = nn.Conv2d(channels, inner, 1, bias=False)
        self.vi = nn.Conv2d(channels, channels, 1, bias=False)
        self.fuse = nn.Sequential(nn.Conv2d(2*channels, channels, 1), nn.ReLU(inplace=True))
        self.ln_s = nn.LayerNorm([channels, 1, 1])
        self.ln_i = nn.LayerNorm([channels, 1, 1])

    def _attend(self, q, k, v):
        B,C,H,W = q.shape
        qf = q.flatten(2).transpose(1,2) # B,HW,Cq
        kf = k.flatten(2).transpose(1,2) # B,HW,Ck
        vf = v.flatten(2).transpose(1,2) # B,HW,Cv
        attn = torch.matmul(qf, kf.transpose(1,2)) / (kf.shape[-1] ** 0.5)
        attn = F.softmax(attn, dim=-1)
        of = torch.matmul(attn, vf).transpose(1,2).reshape(B, -1, H, W)
        return of

    def forward(self, fs, fi):
        # sketch->image
        qs, ks, vs = self.qs(fs), self.ki(fi), self.vi(fi)
        s2i = self._attend(qs, ks, vs)
        # image->sketch
        qi, ki, vi = self.qi(fi), self.ks(fs), self.vs(fs)
        i2s = self._attend(qi, ki, vi)
        # fuse
        fs_out = self.ln_s(self.fuse(torch.cat([fs, i2s], dim=1)) + fs)
        fi_out = self.ln_i(self.fuse(torch.cat([fi, s2i], dim=1)) + fi)
        return fs_out, fi_out
