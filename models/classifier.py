from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class SinusoidalPositionEmbeddings(nn.Module):
    """Ánh xạ timestep t (scalar) thành vector embedding dạng hình sin."""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class _ResBlock(nn.Module):
    """Pre-activation residual block có tích hợp Time Conditioning (FiLM)."""
    def __init__(self, channels: int, out_channels: int, t_embed_dim: int, stride: int = 1, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(8, channels), channels)
        self.conv1 = nn.Conv2d(channels, out_channels, 3, stride=stride, padding=1, bias=False)
        
        self.norm2 = nn.GroupNorm(min(8, out_channels), out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        
        # Projection cho timestep embedding để tạo scale và shift
        self.time_emb_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(t_embed_dim, out_channels * 2)
        )
        
        self.skip = (
            nn.Conv2d(channels, out_channels, 1, stride=stride, bias=False)
            if channels != out_channels or stride != 1
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        # Nhánh chính
        h = self.conv1(F.silu(self.norm1(x)))
        
        # FiLM conditioning ngay giữa block
        time_params = self.time_emb_proj(t_emb)[:, :, None, None]
        scale, shift = time_params.chunk(2, dim=1)
        
        h = self.norm2(h)
        h = h * (1.0 + scale) + shift # Scale & Shift
        
        h = self.conv2(self.dropout(F.silu(h)))
        
        return h + self.skip(x)

class AttentionBlock(nn.Module):
    """Standard Self-Attention cho độ phân giải thấp để bắt Global Context."""
    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.GroupNorm(min(8, channels), channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h).reshape(B, 3, C, H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        
        attn = (q.transpose(-2, -1) @ k) * (C ** -0.5)
        attn = F.softmax(attn, dim=-1)
        
        out = (v @ attn.transpose(-2, -1)).reshape(B, C, H, W)
        return x + self.proj(out)

# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

class BraTSClassifier(nn.Module):
    """
    CNN classifier được tối ưu hóa cho Classifier Guidance trong Flow Matching.
    """
    def __init__(
        self,
        in_channels: int = 4,
        base_channels: int = 64,  # Tăng lên một chút để mạng có đủ capacity
        num_blocks: int = 2,
        t_embed_dim: int = 256,
    ):
        super().__init__()
        self.t_embed_dim = t_embed_dim

        # Timestep embedding: Sinusoidal -> MLP
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(t_embed_dim),
            nn.Linear(t_embed_dim, t_embed_dim),
            nn.SiLU(),
            nn.Linear(t_embed_dim, t_embed_dim),
        )

        self.stem = nn.Conv2d(in_channels, base_channels, 3, padding=1, bias=False)

        # Encoder stages (dùng ModuleList thay vì Sequential vì cần truyền t_emb)
        self.encoder_stages = nn.ModuleList()
        channels = base_channels
        
        for stage_idx in range(4):
            out_ch = channels * 2
            blocks = nn.ModuleList([
                _ResBlock(channels, out_ch, t_embed_dim, stride=2)
            ])
            for _ in range(num_blocks - 1):
                blocks.append(_ResBlock(out_ch, out_ch, t_embed_dim))
            
            self.encoder_stages.append(blocks)
            channels = out_ch

        # Self-Attention ở stage cuối cùng (nơi spatial resolution nhỏ nhất)
        self.mid_attn = AttentionBlock(channels)

        # Classification head
        self.head = nn.Sequential(
            nn.GroupNorm(min(8, channels), channels),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),  
            nn.Flatten(),             
            nn.Dropout(0.2), # Thêm dropout để chống overfit với noise
            nn.Linear(channels, 1),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor | None = None) -> torch.Tensor:
        B = x.shape[0]

        # Timestep embedding
        if t is not None:
            # Scale t (nếu t trong [0, 1]) lên [0, 1000] để Sinusoidal hoạt động tốt nhất
            t_scaled = t.float().view(B) * 1000.0 
            t_emb = self.time_embed(t_scaled)
        else:
            t_emb = torch.zeros(B, self.t_embed_dim, device=x.device, dtype=x.dtype)

        h = self.stem(x)

        for stage_blocks in self.encoder_stages:
            for block in stage_blocks:
                h = block(h, t_emb)

        h = self.mid_attn(h)
        return self.head(h)

def channels_at_stage(base: int, stage: int) -> int:
    return base * (2 ** (stage + 1))