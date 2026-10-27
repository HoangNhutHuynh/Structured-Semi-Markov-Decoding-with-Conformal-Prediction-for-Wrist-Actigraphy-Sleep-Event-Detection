# ----------------------------------------------------------------------------
# Structured Semi-Markov Decoding with Conformal Prediction for
# Wrist-Actigraphy Sleep Event Detection
#
# Author    : Hoang Nhut Huynh
# Copyright : (c) 2026 Hoang Nhut Huynh
# ----------------------------------------------------------------------------
"""B1: Kaggle-style CNN + BiGRU heatmap model (design spec v1.1, section 5).

Input  (B, 7, L=18000) at 5 s; output logits (B, 3, L/12 = 1500) at 1 min:
channel 0 = onset heatmap, 1 = wakeup heatmap, 2 = sleep state.
"""
from __future__ import annotations
import torch
from torch import nn


class ResBlock(nn.Module):
    def __init__(self, ch: int, k: int = 5):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv1d(ch, ch, k, padding=k // 2), nn.BatchNorm1d(ch), nn.GELU(),
            nn.Conv1d(ch, ch, k, padding=k // 2), nn.BatchNorm1d(ch))
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.body(x))


class B1Net(nn.Module):
    def __init__(self, in_ch: int = 7, ch: int = 64, hidden: int = 128, dropout: float = 0.2,
                 pools: tuple[int, ...] = (2, 2, 3)):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(in_ch, ch, 5, padding=2), nn.BatchNorm1d(ch), nn.GELU())
        stages = []
        for p in pools:
            stages += [ResBlock(ch), ResBlock(ch), nn.AvgPool1d(p, stride=p)]
        self.stages = nn.Sequential(*stages)
        self.gru = nn.GRU(ch, hidden, num_layers=2, batch_first=True, bidirectional=True,
                          dropout=dropout)
        self.head = nn.Linear(2 * hidden, 3)
        self.downsample = 1
        for p in pools:
            self.downsample *= p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stages(self.stem(x))            # (B, ch, L/12)
        h, _ = self.gru(h.transpose(1, 2))       # (B, L/12, 2*hidden)
        return self.head(h).transpose(1, 2)      # (B, 3, L/12)


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
