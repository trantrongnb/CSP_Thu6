import torch
from torch.nn import functional as F


def token_matching(support_pos, support_neg, query_pos, query_neg, weights,
                   token_reduction="mean", negative_reduction="separate", negative_weights="opponent"):
    """Index-wise TEAM cosine distances; no T x T alignment.

    The released TEAM code takes each crossed branch's worst competitor
    independently. The design's joint-min extension is an explicit option.
    """
    if token_reduction not in {"mean", "sum"} or negative_reduction not in {"separate", "joint"}:
        raise ValueError("Unknown token/negative reduction")
    if negative_weights not in {"opponent", "candidate", "uniform"}:
        raise ValueError("Unknown exclusive part weighting")
    n, _, m, _ = support_pos.shape

    def similarity(s, q):
        cosine = torch.einsum("crmd,urmd->ucrm", F.normalize(s.float(), dim=-1),
                              F.normalize(q.float(), dim=-1))
        distances = cosine - 1.0
        return distances.mean(-1) if token_reduction == "mean" else distances.sum(-1)

    pos = (similarity(support_pos, query_pos) * weights).sum(-1)
    cross_a, cross_b = similarity(support_pos, query_neg), similarity(support_neg, query_pos)
    if negative_weights == "opponent":
        a = (cross_a * weights).sum(-1)[:, None, :].expand(-1, n, -1)
        b = (cross_b * weights).sum(-1)[:, None, :].expand(-1, n, -1)
    else:
        w = weights if negative_weights == "candidate" else torch.ones_like(weights) / weights.shape[-1]
        a = torch.einsum("uor,cr->uco", cross_a, w)
        b = torch.einsum("uor,cr->uco", cross_b, w)
    same = torch.eye(n, device=pos.device, dtype=torch.bool)[None]
    # Invalid support parts have zero weights. Such classes are filled using
    # the RGB branch by the caller, never offered as fake zero-score opponents.
    no_opponent = weights.sum(-1) <= 0
    excluded = same | no_opponent[None, None, :]
    if negative_reduction == "joint":
        neg = ((a + b) / 2).masked_fill(excluded, float("inf")).amin(-1)
    else:
        neg = (a.masked_fill(excluded, float("inf")).amin(-1)
               + b.masked_fill(excluded, float("inf")).amin(-1)) / 2
    return pos, torch.where(torch.isfinite(neg), neg, torch.zeros_like(neg))
