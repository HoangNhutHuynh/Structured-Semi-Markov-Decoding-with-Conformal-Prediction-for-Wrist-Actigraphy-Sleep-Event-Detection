# ----------------------------------------------------------------------------
# Structured Semi-Markov Decoding with Conformal Prediction for
# Wrist-Actigraphy Sleep Event Detection
#
# Author    : Hoang Nhut Huynh
# Copyright : (c) 2026 Hoang Nhut Huynh
# ----------------------------------------------------------------------------
"""Exactness of the structured decoder against brute-force enumeration."""
import itertools
import math
import torch
from src.models.semimarkov import PairDecoder


def _brute(e_on, e_wk, s, n, dec):
    C = torch.cumsum(s, 0); ld = dec.log_dur_norm()
    scores = {}
    for a in range(n):
        for d in range(dec.dmin, dec.D):
            b = a + d
            if b < n:
                scores[(a, d)] = float((e_on[a] + e_wk[b] + C[b] - C[a] + ld[d]).detach())
    return scores


def test_partition_marginals_map_match_bruteforce():
    torch.manual_seed(0)
    T, D, dmin = 40, 25, 5
    dec = PairDecoder(max_dur=D, min_dur=dmin, init_log_dur=torch.randn(D))
    e_on, e_wk, s = torch.randn(1, T), torch.randn(1, T), torch.randn(1, T)
    n = torch.tensor([33]); s_null = torch.tensor([0.7])
    S = dec.pair_scores(e_on, e_wk, s, n)
    br = _brute(e_on[0], e_wk[0], s[0], 33, dec)
    logZ_b = math.log(sum(math.exp(v) for v in br.values()) + math.exp(0.7))
    assert abs(float(dec.log_partition(S, s_null)) - logZ_b) < 1e-4
    p_null, m_on, m_wk, a_map, d_map, p_map = dec.posterior(S, s_null)
    best = max(br, key=br.get)
    assert (int(a_map), int(d_map)) == best
    on_b = [sum(math.exp(v - logZ_b) for (a, d), v in br.items() if a == t) for t in range(T)]
    wk_b = [sum(math.exp(v - logZ_b) for (a, d), v in br.items() if a + d == t) for t in range(T)]
    assert torch.allclose(m_on[0], torch.tensor(on_b), atol=1e-5)
    assert torch.allclose(m_wk[0], torch.tensor(wk_b), atol=1e-5)
    assert abs(float(p_null) + float(m_on.sum()) - 1) < 1e-5


def test_nll_is_positive_and_gradients_flow():
    torch.manual_seed(1)
    dec = PairDecoder(max_dur=60, min_dur=10, learn_kernel=True)
    e_on = torch.randn(2, 100, requires_grad=True)
    S = dec.pair_scores(e_on, torch.randn(2, 100), torch.randn(2, 100), torch.tensor([100, 80]))
    loss = dec.nll_labelled(S, torch.zeros(2), torch.tensor([20, 30]), torch.tensor([60, 70])).mean()
    loss = loss + dec.nll_null(S, torch.zeros(2)).mean()
    loss.backward()
    assert float(loss) > 0 and e_on.grad.abs().sum() > 0 and dec.k_shift.grad is not None
