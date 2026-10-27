from torch.nn import functional as F

from .hard_pair import hard_pair_motion_loss


def episodic_loss(output, labels, exclusive=True, k_shot=1, hard_pair_lambda=0.0,
                  hard_pair_margin=0.2, weight_kl_lambda=0.0):
    if hard_pair_lambda < 0 or weight_kl_lambda < 0:
        raise ValueError("Loss coefficients must be nonnegative")
    loss_pos = F.cross_entropy(output["logits_pos"], labels)
    loss_neg = F.cross_entropy(output["logits_neg"], labels) if exclusive else loss_pos.new_zeros(())
    hard = hard_pair_motion_loss(output, labels, k_shot, hard_pair_margin) if hard_pair_lambda else loss_pos.new_zeros(())
    w = output["part_weights"]
    valid = w > 0
    uniform_count = valid.sum(-1, keepdim=True).clamp_min(1)
    kl = (w * (w.clamp_min(1e-6) * uniform_count).clamp_min(1e-6).log()).sum(-1).mean()
    total = loss_pos + loss_neg + hard_pair_lambda * hard + weight_kl_lambda * kl
    return total, {"positive": loss_pos, "exclusive": loss_neg, "hard_pair": hard, "weight_kl": kl}
