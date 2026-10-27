# ----------------------------------------------------------------------------
# Structured Semi-Markov Decoding with Conformal Prediction for
# Wrist-Actigraphy Sleep Event Detection
#
# Author    : Hoang Nhut Huynh
# Copyright : (c) 2026 Hoang Nhut Huynh
# ----------------------------------------------------------------------------
"""Phase 2 flagship: structured one-pair-per-night decoder (exact, differentiable).

For one window with n valid 1-min bins and encoder scores e_on(t), e_wk(t), s(t):
    S(a, d) = e_on(a) + e_wk(a+d) + [C(a+d) - C(a)] + log p_dur(d),   C = prefix sum of s,
for onset bin a, duration d in [d_min, D), wake bin a+d < n; plus one "no valid window" state with
score S_null. The space has only O(n*D) pairs, so the partition function, marginals and MAP are
computed EXACTLY by dense (T x D) tensors (a semi-Markov chain wake->sleep->wake with a single
sleep segment); no approximation and no NMS.

Training (labelled window, labels a*, b*):  NLL = log Z - log sum_{a,d} k_on(a-a*) k_wk(a+d-b*) e^S
  with k = normalised Gaussian label-noise kernels (optionally learnable shift/width per event).
Proxy-negative window (>=90% non-wear, no label):  NLL = log Z - S_null  (learn to abstain).
Unlabelled windows are missing data: no loss.
"""
from __future__ import annotations
import math
import torch
from torch import nn
import torch.nn.functional as F

NEG = -1e4


class PairDecoder(nn.Module):
    def __init__(self, max_dur: int = 1200, min_dur: int = 30, init_log_dur: torch.Tensor | None = None,
                 kernel_sigma: float = 1.5, learn_kernel: bool = False):
        super().__init__()
        self.D, self.dmin = max_dur, min_dur
        init = torch.zeros(max_dur) if init_log_dur is None else init_log_dur.float().clone()
        self.log_dur = nn.Parameter(init)                    # learnable duration prior (bins)
        self.k_shift = nn.Parameter(torch.zeros(2), requires_grad=learn_kernel)
        self.k_logsig = nn.Parameter(torch.full((2,), math.log(kernel_sigma)), requires_grad=learn_kernel)

    # ---- core tensor: S[b, a, d] -------------------------------------------------------------
    def pair_scores(self, e_on, e_wk, s, n):
        """e_on, e_wk, s: (B, T); n: (B,) valid lengths. Returns (B, T, D) with NEG on invalid."""
        Bsz, T = e_on.shape
        C = torch.cumsum(s, dim=1)
        A = e_on - C                                          # onset term
        Bt = e_wk + C                                         # wake term (at a+d)
        Bp = F.pad(Bt, (0, self.D), value=NEG)                # (B, T+D)
        Bunf = Bp.unfold(1, self.D, 1)[:, :T, :]              # Bunf[b,a,d] = Bt[b,a+d]
        S = A.unsqueeze(-1) + Bunf + self.log_dur_norm().view(1, 1, -1)
        a = torch.arange(T, device=S.device).view(1, T, 1)
        d = torch.arange(self.D, device=S.device).view(1, 1, -1)
        valid = (d >= self.dmin) & (a + d < n.view(-1, 1, 1))
        return S.masked_fill(~valid, NEG)

    def log_dur_norm(self):
        x = self.log_dur.masked_fill(torch.arange(self.D, device=self.log_dur.device) < self.dmin, NEG)
        return x - torch.logsumexp(x, 0)

    def log_partition(self, S, s_null):
        flat = S.flatten(1)
        return torch.logsumexp(torch.cat([flat, s_null.view(-1, 1)], 1), 1)

    def kernel(self, centre, T, which):
        """(B, T) normalised log Gaussian kernel around centre (bins) for event which (0/1)."""
        t = torch.arange(T, device=centre.device, dtype=torch.float32).view(1, -1)
        mu = centre.view(-1, 1).float() + self.k_shift[which]
        sig = self.k_logsig[which].exp()
        lk = -0.5 * ((t - mu) / sig) ** 2
        return lk - torch.logsumexp(lk, 1, keepdim=True)

    def nll_labelled(self, S, s_null, a_star, b_star):
        Bsz, T, D = S.shape
        lk_on = self.kernel(a_star, T, 0)                     # (B, T)
        lk_wk = self.kernel(b_star, T, 1)
        lk_wk_unf = F.pad(lk_wk, (0, D), value=NEG).unfold(1, D, 1)[:, :T, :]
        num = torch.logsumexp((S + lk_on.unsqueeze(-1) + lk_wk_unf).flatten(1), 1)
        return self.log_partition(S, s_null) - num

    def nll_null(self, S, s_null):
        return self.log_partition(S, s_null) - s_null

    # ---- inference ---------------------------------------------------------------------------
    @torch.no_grad()
    def posterior(self, S, s_null):
        """p_null (B,), onset marginal (B,T), wake marginal (B,T), MAP (a, d) and its prob."""
        Bsz, T, D = S.shape
        logZ = self.log_partition(S, s_null)
        P = torch.exp(S - logZ.view(-1, 1, 1))                # (B,T,D)
        p_null = torch.exp(s_null - logZ)
        m_on = P.sum(2)
        idx = (torch.arange(T, device=S.device).view(1, T, 1) +
               torch.arange(D, device=S.device).view(1, 1, D)).expand(Bsz, T, D)
        m_wk = torch.zeros(Bsz, T + D, device=S.device).scatter_add_(1, idx.reshape(Bsz, -1),
                                                                     P.reshape(Bsz, -1))[:, :T]
        best = S.flatten(1).argmax(1)
        a_map, d_map = best // D, best % D
        p_map = torch.exp(S.flatten(1).gather(1, best.view(-1, 1)).squeeze(1) - logZ)
        return p_null, m_on, m_wk, a_map, d_map, p_map


class FlagshipNet(nn.Module):
    """B1 backbone + per-bin heads (e_on, e_wk, s) + window-level null head + PairDecoder."""

    def __init__(self, backbone: nn.Module, decoder: PairDecoder, hidden: int = 256):
        super().__init__()
        self.backbone = backbone                              # B1Net without its head usage
        self.null_head = nn.Sequential(nn.Linear(hidden + 1, 64), nn.GELU(), nn.Linear(64, 1))
        self.decoder = decoder

    def encode(self, x, mask):
        bb = self.backbone
        h = bb.stages(bb.stem(x))
        h, _ = bb.gru(h.transpose(1, 2))                      # (B, M, 2H)
        out = bb.head(h)                                      # (B, M, 3)
        m = mask.unsqueeze(-1).float()
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1)
        nonwear = x[:, 6].reshape(x.shape[0], -1, 12).mean(-1)   # per-bin non-wear fraction
        nw = (nonwear * mask).sum(1, keepdim=True) / mask.sum(1, keepdim=True).clamp(min=1)
        s_null = self.null_head(torch.cat([pooled, nw], 1)).squeeze(1)
        return out[..., 0], out[..., 1], out[..., 2], s_null

    def forward(self, x, mask):
        e_on, e_wk, s, s_null = self.encode(x, mask)
        e_on, e_wk, s, s_null = e_on.float(), e_wk.float(), s.float(), s_null.float()
        self.last_heads = (e_on, e_wk)                        # raw per-bin logits (aux heatmap loss)
        n = mask.sum(1)
        neg = ~mask
        e_on = e_on.masked_fill(neg, NEG); e_wk = e_wk.masked_fill(neg, NEG); s = s.masked_fill(neg, 0.0)
        S = self.decoder.pair_scores(e_on, e_wk, s, n)
        return S, s_null, s
