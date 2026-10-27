import torch
from torch.nn import functional as F


def adapt_support(aggregator, context, positive, negative, prior, valid=None, enabled=True):
    """Exact TEAM similarity-scaled competitor adaptation, independently per part.

    Inputs [N,R,M,D]. Apply MLP/ReLU to EACH competitor-conditioned reference,
    THEN average competitors (the official code's order of operations).
    """
    if not enabled:
        return positive, negative
    n = len(context)
    if n < 2:
        raise ValueError("TEAM adaptation needs at least two support classes")
    if valid is None:
        valid = torch.ones(context.shape[:2], device=context.device, dtype=torch.bool)
    not_self = ~torch.eye(n, device=context.device, dtype=torch.bool)
    eligible = not_self[:, :, None] & valid[:, None, :] & valid[None, :, :]
    displacement = context[None, :] - context[:, None]
    sp = F.cosine_similarity(positive[:, None], positive[None, :], dim=-1)
    sn = F.cosine_similarity(negative[:, None], negative[None, :], dim=-1)
    # Avoid materializing a separate attention pass: contexts already contain CA.
    pos = prior + context[:, None] - sp[..., None] * displacement
    neg = prior - context[:, None] + sn[..., None] * displacement
    pos, neg = F.relu(pos + aggregator.mlp(pos)), F.relu(neg + aggregator.mlp(neg))
    weights = eligible[..., None, None].to(context.dtype)
    count = weights.sum(1).clamp_min(1)
    pos, neg = (pos * weights).sum(1) / count, (neg * weights).sum(1) / count
    has_competitor = eligible.any(1)[..., None, None]
    return torch.where(has_competitor, pos, positive), torch.where(has_competitor, neg, negative)
