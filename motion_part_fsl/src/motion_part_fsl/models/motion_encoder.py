import torch
from torch import nn


# 2D pose, timestamp-scaled velocity, acceleration, and wrist-relative handshape.
HAND_DIM = 21 * 2 * 4 + 3  # plus pose/velocity/acceleration validity
RELATION_DIM = 2 + 1 + 2 + 2 + 3  # delta, distance, velocity, palm directions, masks


class MotionEncoder(nn.Module):
    def __init__(self, dimension):
        super().__init__()

        def mlp(inputs):
            return nn.Sequential(nn.Linear(inputs, dimension), nn.ReLU(),
                                 nn.Linear(dimension, dimension), nn.LayerNorm(dimension))

        self.hands = nn.ModuleList([mlp(HAND_DIM), mlp(HAND_DIM)])
        self.relation = mlp(RELATION_DIM)

    def forward(self, hands, relation, quality):
        parts = [self.hands[r](hands[:, r]) for r in range(2)]
        parts.append(self.relation(relation))
        motion = torch.stack(parts, dim=1)
        # Linear biases must not turn missing landmarks into valid evidence.
        return motion * quality.unsqueeze(-1)
