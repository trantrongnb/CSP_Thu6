import torch
from torch.nn import functional as F


def hard_pair_motion_loss(output, query_labels, k_shot, margin=0.2):
    """Appearance mines support negatives; motion-aware embeddings get gradients."""
    sa, qa = output["support_appearance"].detach(), output["query_appearance"].detach()
    n = sa.shape[0] // k_shot
    support_labels = torch.arange(n, device=sa.device).repeat_interleave(k_shot)
    appearance_sim = F.normalize(qa, dim=-1) @ F.normalize(sa, dim=-1).T
    negative = appearance_sim.masked_fill(support_labels[None] == query_labels[:, None], -2).argmax(-1)
    sm, qm = output["support_motion_embedding"], output["query_motion_embedding"]
    positive = sm.reshape(n, k_shot, -1).mean(1)[query_labels]
    sim_pos = F.cosine_similarity(qm, positive, dim=-1)
    sim_neg = F.cosine_similarity(qm, sm[negative], dim=-1)
    return F.relu(margin + sim_neg - sim_pos).mean()
