import torch
from torch import nn
from torch.nn import functional as F


class SupportPartWeights(nn.Module):
    def __init__(self, parts=3, mode="adaptive", temperature=0.2,
                 reliability_lambda=0.5, uniform_shrinkage=0.2, detach=False):
        super().__init__()
        if mode not in {"adaptive", "uniform", "learned", "episode"}:
            raise ValueError(f"Unknown part weight mode: {mode}")
        if temperature <= 0 or reliability_lambda < 0 or not 0 <= uniform_shrinkage <= 1:
            raise ValueError("Invalid EAPM temperature, reliability or shrinkage")
        self.mode, self.temperature = mode, temperature
        self.reliability_lambda, self.shrinkage, self.detach = reliability_lambda, uniform_shrinkage, detach
        self.global_logits = nn.Parameter(torch.zeros(parts), requires_grad=mode == "learned")

    def forward(self, support_positive, reliability):
        n, r, m, _ = support_positive.shape
        if n < 2 or reliability.shape != (n, r):
            raise ValueError("EAPM needs [N,R,M,D], [N,R] and N>=2")
        valid = reliability > 0
        x = F.normalize(support_positive.float(), dim=-1)
        sim = torch.einsum("crmd,ormd->cor", x, x) / m
        competitors = valid[:, None] & valid[None, :]
        competitors &= ~torch.eye(n, device=x.device, dtype=torch.bool)[:, :, None]
        nearest = sim.masked_fill(~competitors, -2.0).amax(1)
        # A part without a valid opponent has no estimated separation.
        eligible = valid & competitors.any(1)
        discr = torch.where(eligible, (1 - nearest).clamp(0, 2), torch.zeros_like(nearest))
        if self.mode in {"uniform", "learned"}:
            eligible = valid
        uniform = eligible.float() / eligible.sum(-1, keepdim=True).clamp_min(1)
        if self.mode == "uniform":
            return uniform, discr
        logits = discr / self.temperature
        logits = logits + self.reliability_lambda * reliability.float().clamp_min(1e-6).log()
        if self.mode == "learned":
            logits = self.global_logits.expand(n, -1)
        elif self.mode == "episode":
            pooled = (logits * eligible).sum(0) / eligible.sum(0).clamp_min(1)
            logits = pooled.expand(n, -1)
        raw = F.softmax(logits.masked_fill(~eligible, -1e4), -1) * eligible
        raw = raw / raw.sum(-1, keepdim=True).clamp_min(1e-6)
        weights = (1 - self.shrinkage) * raw + self.shrinkage * uniform
        return (weights.detach() if self.detach else weights), discr
