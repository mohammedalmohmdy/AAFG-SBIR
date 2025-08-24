import torch, torch.nn as nn, torch.nn.functional as F
from torchvision import models
from .attention import SelfAttention2d, CrossAttention2d

class ProjectionHead(nn.Module):
    def __init__(self, in_dim, embed_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim), nn.ReLU(inplace=True),
            nn.Linear(in_dim, embed_dim)
        )
    def forward(self, x):
        x = self.net(x)
        x = F.normalize(x, p=2, dim=-1)
        return x

class AAFG_SBiR(nn.Module):
    def __init__(self, backbone='resnet50', embed_dim=512, attn_reduction=1):
        super().__init__()
        assert backbone=='resnet50', 'Only resnet50 implemented for simplicity.'
        base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.stem = nn.Sequential(*(list(base.children())[:5]))  # up to layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.c5 = 2048
        self.self_sketch = SelfAttention2d(self.c5, reduction=attn_reduction)
        self.self_image  = SelfAttention2d(self.c5, reduction=attn_reduction)
        self.cross = CrossAttention2d(self.c5, reduction=attn_reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = ProjectionHead(self.c5, embed_dim)

    def encode(self, x):
        x = self.stem(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def forward(self, sketch, image):
        fs = self.encode(sketch)
        fi = self.encode(image)
        fs = self.self_sketch(fs)
        fi = self.self_image(fi)
        fs, fi = self.cross(fs, fi)
        zs = self.head(self.pool(fs).flatten(1))
        zi = self.head(self.pool(fi).flatten(1))
        return zs, zi
