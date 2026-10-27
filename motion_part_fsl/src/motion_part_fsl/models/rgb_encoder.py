import torch
from torch import nn


class RGBEncoder(nn.Module):
    """Frame encoder. Frozen BN statistics keep support independent of queries."""

    def __init__(self, backbone="resnet50", feature_dim=256, pretrained=True,
                 freeze_backbone=False, frame_batch_size=16, gradient_checkpointing=False):
        super().__init__()
        if frame_batch_size < 1:
            raise ValueError("frame_batch_size must be positive")
        self.frame_batch_size, self.gradient_checkpointing = frame_batch_size, gradient_checkpointing
        if backbone == "tiny":
            if pretrained:
                raise ValueError("The tiny smoke-test encoder has no pretrained weights")
            self.network = nn.Sequential(
                nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.GroupNorm(4, 16), nn.ReLU(),
                nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.GroupNorm(4, 32), nn.ReLU(),
                nn.AdaptiveAvgPool2d(1), nn.Flatten())
            output_dim = 32
        elif backbone == "resnet50":
            from torchvision.models import ResNet50_Weights, resnet50
            network = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
            output_dim = network.fc.in_features
            network.fc = nn.Identity()
            self.network = network
        elif backbone == "vit_b_16":
            from torchvision.models import ViT_B_16_Weights, vit_b_16
            network = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None)
            output_dim = network.hidden_dim
            network.heads = nn.Identity()
            self.network = network
        else:
            raise ValueError(f"Unknown RGB backbone: {backbone}")
        self.projection = nn.Identity() if output_dim == feature_dim else nn.Linear(output_dim, feature_dim)
        if freeze_backbone:
            self.network.requires_grad_(False)
        # The torchvision ViT has dropout=0 by default. No cross-video normalization.
        self.train()

    def train(self, mode=True):
        super().train(mode)
        for module in self.network.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
        return self

    def forward(self, rgb):
        batch, frames = rgb.shape[:2]
        chunks = []
        for chunk in rgb.flatten(0, 1).split(self.frame_batch_size):
            if self.gradient_checkpointing and self.training and any(p.requires_grad for p in self.network.parameters()):
                from torch.utils.checkpoint import checkpoint
                output = checkpoint(self.network, chunk, use_reentrant=False)
            else:
                output = self.network(chunk)
            chunks.append(self.projection(output))
        features = torch.cat(chunks)
        return features.reshape(batch, frames, -1)
