import math

import torch
from torch import nn
from torch.nn import functional as F


class PatternAggregator(nn.Module):
    """TEAM's shared pattern bank, plus optional MGPR keys/values/saliency.

    The RGB path retains the original LN, PE, sqrt(d_k), dropout, shared MLP,
    and ReLU. Contexts are exposed so support adaptation runs before the MLP.
    """

    def __init__(self, dimension=256, pattern_tokens=16, frames=8, heads=8,
                 dropout=0.1, attention_dropout=0.1, lambda_motion=1.0):
        super().__init__()
        if dimension < 2 or dimension % heads or frames < 2 or pattern_tokens < 1:
            raise ValueError("Require D divisible by heads, T>=2 and M>=1")
        self.dimension, self.frames, self.heads = dimension, frames, heads
        self.patterns = nn.Parameter(torch.randn(1, pattern_tokens, dimension))
        self.part_embeddings = nn.Parameter(torch.zeros(3, dimension))
        nn.init.normal_(self.part_embeddings, std=0.02)
        self.motion_keys = nn.Linear(dimension, dimension, bias=False)
        self.saliency = nn.Parameter(torch.randn(3, pattern_tokens, dimension) / math.sqrt(dimension))
        self.concat_projection = nn.Linear(3 * dimension, dimension, bias=False)
        self.lambda_motion = float(lambda_motion)
        self.lambda_bias = nn.Parameter(torch.tensor(0.1))
        self.ln = nn.LayerNorm(dimension)
        self.motion_norm = nn.LayerNorm(dimension)
        self.mlp = nn.Sequential(nn.Linear(dimension, dimension), nn.ReLU(), nn.Linear(dimension, dimension))
        self.pe_dropout = nn.Dropout(dropout)
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.output_dropout = nn.Dropout(dropout)
        position = torch.arange(frames).float().unsqueeze(1)
        frequencies = torch.exp(torch.arange(0, dimension, 2).float() * (-math.log(10000.) / dimension))
        pe = torch.zeros(frames, dimension)
        pe[:, 0::2] = torch.sin(position * frequencies) * 0.1
        pe[:, 1::2] = torch.cos(position * frequencies[:dimension // 2]) * 0.1
        self.register_buffer("position", pe)

    def forward(self, appearance, motion, quality, frame_valid, mode="full", parts=(0, 1, 2)):
        if mode not in {"rgb", "concat", "full", "bias", "key", "value", "motion_off"}:
            raise ValueError(f"Unknown aggregation mode: {mode}")
        if appearance.shape[1] != self.frames or not frame_valid.any(-1).all():
            raise ValueError("RGB must have the configured frame count and at least one valid frame")
        batch = len(appearance)
        if mode in {"rgb", "concat"}:
            visual = appearance[:, None]
            if mode == "concat":
                fused = self.concat_projection(motion.permute(0, 2, 1, 3).flatten(2))
                visual = visual + self.lambda_motion * fused[:, None]
            prior = self.patterns
            valid = frame_valid[:, None]
            values = self.ln(self.pe_dropout(visual + self.position))
            keys = values
            bias = 0.0
        else:
            embeddings = self.part_embeddings[list(parts)]
            prior = self.patterns + embeddings[:, None]
            motion, quality = motion[:, list(parts)], quality[:, list(parts)]
            visual = appearance[:, None] + embeddings[None, :, None]
            fused = visual
            if mode in {"full", "value"}:
                fused = fused + self.lambda_motion * motion
            values = self.ln(self.pe_dropout(fused + self.position))
            # Value-only explicitly holds keys RGB-only, isolating guidance.
            keys = self.ln(self.pe_dropout(visual + self.position))
            if mode in {"full", "key"}:
                keys = keys + self.lambda_motion * self.motion_keys(motion)
            bias = 0.0
            guided = mode in {"full", "key", "bias"} and self.lambda_motion != 0
            if guided:
                saliency = torch.einsum("rmd,brtd->brmt", self.saliency[list(parts)], self.motion_norm(motion))
                bias = self.lambda_motion * self.lambda_bias * torch.tanh(saliency)
                bias = bias * (quality > 0)[:, :, None]
                bias = bias + torch.log(quality.clamp_min(1e-6))[:, :, None]
                has_landmarks = (quality > 0).any(-1, keepdim=True)
                # All-invalid parts attend valid RGB frames, never all -inf.
                valid = torch.where(has_landmarks, quality > 0, frame_valid[:, None])
                bias = torch.where(has_landmarks[:, :, None], bias, torch.zeros_like(bias))
            else:
                valid = frame_valid[:, None].expand(batch, len(parts), -1)
        logits = torch.einsum("rmd,brtd->brmt", self.ln(prior), keys) / math.sqrt(self.dimension / self.heads)
        logits = logits + bias
        logits = logits.masked_fill(~valid[:, :, None], float("-inf"))
        # Softmax in FP32 is important for AMP and low-confidence landmarks.
        attention = torch.softmax(logits.float(), dim=-1).to(values.dtype)
        context = torch.einsum("brmt,brtd->brmd", self.attention_dropout(attention), values)
        context = self.output_dropout(context)
        positive, negative = self.tokens(context, prior)
        return {"context": context, "positive": positive, "negative": negative,
                "attention": attention, "prior": prior}

    def tokens(self, context, prior):
        positive, negative = prior + context, prior - context
        return F.relu(positive + self.mlp(positive)), F.relu(negative + self.mlp(negative))
