import torch
from torch import nn

from .rgb_encoder import RGBEncoder
from .motion_encoder import MotionEncoder
from .motion_guided_patterns import PatternAggregator
from .support_adaptation import adapt_support
from .part_weighting import SupportPartWeights
from .matching import token_matching


VARIANTS = {
    "A0": ("rgb", "uniform", True),
    "A1": ("concat", "uniform", True),
    "A2": ("full", "uniform", True),
    "A3": ("value", "uniform", True),
    "A4": ("value", "adaptive", True),
    "A5": ("full", "adaptive", True),
    "A6": ("full", "adaptive", True),
    "A7": ("full", "adaptive", False),
}


class MotionPartFSL(nn.Module):
    """Support-only episodic inference. Query labels never enter forward()."""

    def __init__(self, backbone="resnet50", feature_dim=256, pattern_tokens=16,
                 frames=8, heads=8, pretrained=True, freeze_backbone=False,
                 frame_batch_size=16, gradient_checkpointing=False,
                 variant="A5", parts=(0, 1, 2), dropout=0.1, attention_dropout=0.1,
                 lambda_motion=1.0, guidance_mode=None, support_aggregation="frames",
                 support_adaptation=True, token_reduction="mean", negative_reduction="separate",
                 negative_weights="opponent", temperature=0.2, gamma_positive=1.0,
                 gamma_negative=1.0, part_weights=None):
        super().__init__()
        if variant not in VARIANTS or not parts or len(set(parts)) != len(parts) or not set(parts) <= {0, 1, 2}:
            raise ValueError("Invalid variant or parts (0=left, 1=right, 2=relation)")
        if support_aggregation not in {"frames", "tokens"} or temperature <= 0:
            raise ValueError("Invalid support aggregation or score temperature")
        self.variant, self.parts = variant, tuple(parts)
        mode, weight_mode, self.exclusive = VARIANTS[variant]
        self.mode = guidance_mode or mode
        if mode in {"rgb", "concat"} and guidance_mode not in {None, mode}:
            raise ValueError("A0/A1 have fixed RGB/concat modes")
        self.support_aggregation, self.support_adaptation = support_aggregation, support_adaptation
        self.temperature, self.gamma_positive, self.gamma_negative = temperature, gamma_positive, gamma_negative
        self.matching_options = dict(token_reduction=token_reduction, negative_reduction=negative_reduction,
                                     negative_weights=negative_weights)
        self.rgb = RGBEncoder(backbone, feature_dim, pretrained, freeze_backbone,
                              frame_batch_size, gradient_checkpointing)
        self.motion = MotionEncoder(feature_dim)
        self.pattern = PatternAggregator(feature_dim, pattern_tokens, frames, heads,
                                         dropout, attention_dropout, lambda_motion)
        options = {"mode": weight_mode, **(part_weights or {})}
        self.weighting = SupportPartWeights(len(parts), **options)
        # Freeze modules excluded by an ablation so parameter reports are honest.
        if self.mode == "rgb":
            self.motion.requires_grad_(False)
        if self.mode in {"rgb", "concat"}:
            self.pattern.part_embeddings.requires_grad_(False)
        if self.mode not in {"full", "key"} or lambda_motion == 0:
            self.pattern.motion_keys.requires_grad_(False)
        if self.mode not in {"full", "key", "bias"} or lambda_motion == 0:
            self.pattern.saliency.requires_grad_(False)
            self.pattern.motion_norm.requires_grad_(False)
            self.pattern.lambda_bias.requires_grad_(False)
        if self.mode != "concat":
            self.pattern.concat_projection.requires_grad_(False)

    def _encode(self, batch):
        appearance = self.rgb(batch["rgb"])
        quality = batch["quality"] * batch["frame_valid"][:, None]
        if self.mode == "rgb":
            motion = appearance.new_zeros(len(appearance), 3, appearance.shape[1], appearance.shape[2])
        else:
            motion = self.motion(batch["hand_features"], batch["relation_features"], quality)
        return {"appearance": appearance, "motion": motion, "quality": quality,
                "frame_valid": batch["frame_valid"].bool()}

    def _aggregate(self, features, mode):
        return self.pattern(**features, mode=mode, parts=self.parts)

    def _support_references(self, encoded, n, k, mode):
        if self.support_aggregation == "frames":
            valid = encoded["frame_valid"].reshape(n, k, -1)
            frame_count = valid.sum(1).clamp_min(1)
            appearance = encoded["appearance"].reshape(n, k, *encoded["appearance"].shape[1:])
            appearance = (appearance * valid[..., None]).sum(1) / frame_count[..., None]
            motion = encoded["motion"].reshape(n, k, *encoded["motion"].shape[1:])
            motion = (motion * valid[:, :, None, :, None]).sum(1) / frame_count[:, None, :, None]
            quality = encoded["quality"].reshape(n, k, *encoded["quality"].shape[1:]).sum(1)
            quality = quality / frame_count[:, None]
            aggregate = self._aggregate(dict(appearance=appearance, motion=motion, quality=quality,
                                              frame_valid=valid.any(1)), mode)
        else:
            all_tokens = self._aggregate(encoded, mode)
            aggregate = {key: value.reshape(n, k, *value.shape[1:]).mean(1)
                         for key, value in all_tokens.items() if key != "prior"}
            aggregate["prior"] = all_tokens["prior"]
        return aggregate

    def _adapt(self, support, valid=None):
        return adapt_support(self.pattern, support["context"], support["positive"],
                             support["negative"], support["prior"], valid, self.support_adaptation)

    def forward(self, support, query):
        n, k = support["rgb"].shape[:2]
        if n < 2 or k < 1:
            raise ValueError("Expected support [N,K,T,3,H,W] with N>=2, K>=1")
        flat_support = {key: value.flatten(0, 1) for key, value in support.items()}
        sf, qf = self._encode(flat_support), self._encode(query)
        s = self._support_references(sf, n, k, self.mode)
        q = self._aggregate(qf, self.mode)
        coverage = sf["quality"].sum(-1) / sf["frame_valid"].sum(-1).clamp_min(1)[:, None]
        coverage = coverage.reshape(n, k, 3).mean(1)[:, list(self.parts)]
        if self.mode in {"rgb", "concat"}:
            weights = sf["appearance"].new_ones(n, 1)
            separation = weights.new_zeros(n, 1)
            sp, sn = self._adapt(s)
        else:
            sp, sn = self._adapt(s, coverage > 0)
            weights, separation = self.weighting(sp, coverage)
        pos, neg = token_matching(sp, sn, q["positive"], q["negative"], weights, **self.matching_options)
        if self.mode not in {"rgb", "concat"}:
            # A complete RGB TEAM path supplies fallback scores. This has no
            # extra backbone or pattern-bank parameters and no query statistics.
            rgb_s = self._support_references(sf, n, k, "rgb")
            rgb_q = self._aggregate(qf, "rgb")
            rsp, rsn = self._adapt(rgb_s)
            rgb_pos, rgb_neg = token_matching(rsp, rsn, rgb_q["positive"], rgb_q["negative"],
                                              torch.ones_like(weights[:, :1]), **self.matching_options)
            fallback = (weights.sum(-1) == 0)[None] | ~(qf["quality"][:, list(self.parts)] > 0).any((1, 2))[:, None]
            pos, neg = torch.where(fallback, rgb_pos, pos), torch.where(fallback, rgb_neg, neg)
        logits_pos, logits_neg = pos / self.temperature, neg / self.temperature
        logits = self.gamma_positive * logits_pos
        if self.exclusive:
            logits = logits + self.gamma_negative * logits_neg
        result = {"logits": logits, "logits_pos": logits_pos, "logits_neg": logits_neg,
                "part_weights": weights, "part_separation": separation, "support_reliability": coverage,
                "query_attention": q["attention"],
                "support_appearance": self._pool_rgb(sf), "query_appearance": self._pool_rgb(qf)}
        if self.variant == "A6" and self.training:
            result["support_motion_embedding"] = self._aggregate(sf, self.mode)["positive"].mean((1, 2))
            result["query_motion_embedding"] = q["positive"].mean((1, 2))
        return result

    @staticmethod
    def _pool_rgb(encoded):
        mask = encoded["frame_valid"]
        return (encoded["appearance"] * mask[..., None]).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
