"""SSFTT: Spectral-Spatial Feature Tokenization Transformer for HSI classification.

Reference: Sun, Zhao, Zheng & Wu, "Spectral-Spatial Feature Tokenization
Transformer for Hyperspectral Image Classification," IEEE TGRS 2022
(https://github.com/zgr6010/HSI_SSFTT).

Input: (B, 1, pca_components, patch_size, patch_size) — 3D PCA-reduced patch.
Output: (B, n_classes) — class logits.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class _Residual(nn.Module):
    def __init__(self, fn: nn.Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        return self.fn(x, **kwargs) + x


class _LayerNormalize(nn.Module):
    def __init__(self, dim: int, fn: nn.Module):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        return self.fn(self.norm(x), **kwargs)


class _MLPBlock(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class _Attention(nn.Module):
    def __init__(self, dim: int, heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.to_qkv = nn.Linear(dim, dim * 3, bias=True)
        self.nn1 = nn.Linear(dim, dim)
        self.do1 = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        b, n, d = x.shape
        h = self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = (t.reshape(b, n, h, d // h).permute(0, 2, 1, 3) for t in qkv)
        dots = torch.einsum("bhid,bhjd->bhij", q, k) * self.scale
        attn = dots.softmax(dim=-1)
        out = torch.einsum("bhij,bhjd->bhid", attn, v)
        out = out.permute(0, 2, 1, 3).reshape(b, n, d)
        return self.do1(self.nn1(out))


class _Transformer(nn.Module):
    def __init__(self, dim: int, depth: int, heads: int, mlp_dim: int, dropout: float):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleList([
                _Residual(_LayerNormalize(dim, _Attention(dim, heads=heads, dropout=dropout))),
                _Residual(_LayerNormalize(dim, _MLPBlock(dim, mlp_dim, dropout=dropout))),
            ])
            for _ in range(depth)
        ])

    def forward(self, x: Tensor) -> Tensor:
        for attention, mlp in self.layers:
            x = attention(x)
            x = mlp(x)
        return x


class SSFTTnet(nn.Module):
    """Spectral-Spatial Feature Tokenization Transformer (Sun et al., IEEE TGRS 2022)."""

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 9,
        pca_components: int = 30,
        patch_size: int = 13,
        num_tokens: int = 4,
        dim: int = 64,
        depth: int = 1,
        heads: int = 8,
        mlp_dim: int = 8,
        dropout: float = 0.1,
        emb_dropout: float = 0.1,
    ):
        super().__init__()
        self.pca_components = pca_components
        self.patch_size = patch_size
        self.L = num_tokens
        self.cT = dim

        self.conv3d_features = nn.Sequential(
            nn.Conv3d(in_channels, out_channels=8, kernel_size=(3, 3, 3)),
            nn.BatchNorm3d(8),
            nn.ReLU(),
        )

        conv2d_in = 8 * (pca_components - 2)
        self.conv2d_features = nn.Sequential(
            nn.Conv2d(in_channels=conv2d_in, out_channels=64, kernel_size=(3, 3)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        # Gaussian weighted feature tokenizer parameters
        self.token_wA = nn.Parameter(torch.empty(1, self.L, 64), requires_grad=True)
        nn.init.xavier_normal_(self.token_wA)
        self.token_wV = nn.Parameter(torch.empty(1, 64, self.cT), requires_grad=True)
        nn.init.xavier_normal_(self.token_wV)

        self.pos_embedding = nn.Parameter(torch.empty(1, num_tokens + 1, dim))
        nn.init.normal_(self.pos_embedding, std=0.02)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = _Transformer(dim, depth, heads, mlp_dim, dropout)
        self.to_cls_token = nn.Identity()

        self.nn1 = nn.Linear(dim, num_classes)
        nn.init.xavier_uniform_(self.nn1.weight)
        nn.init.normal_(self.nn1.bias, std=1e-6)

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, 1, pca_components, patch_size, patch_size) -> logits (B, num_classes)."""
        x = self.conv3d_features(x)
        b, c, d, h, w = x.shape
        x = x.reshape(b, c * d, h, w)
        x = self.conv2d_features(x)
        x = x.flatten(start_dim=2).transpose(1, 2)  # (B, H'*W', 64)

        wa = self.token_wA.transpose(1, 2)  # (1, 64, L)
        A = torch.einsum("bij,bjk->bik", x, wa)
        A = A.transpose(1, 2).softmax(dim=-1)  # (B, L, H'*W')
        VV = torch.einsum("bij,bjk->bik", x, self.token_wV)  # (B, H'*W', cT)
        T = torch.einsum("bij,bjk->bik", A, VV)  # (B, L, cT)

        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, T), dim=1)
        x = x + self.pos_embedding
        x = self.dropout(x)
        x = self.transformer(x)
        x = self.to_cls_token(x[:, 0])
        return self.nn1(x)

