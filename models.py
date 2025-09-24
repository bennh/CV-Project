# models.py

import torch
import torch.nn as nn
import torchvision.models as models


class PoseNet(nn.Module):
    """
    PoseNet 实现: 使用 ResNet backbone, 输出 3D translation + 4D quaternion
    """
    def __init__(self, backbone="resnet34", pretrained=True):
        super().__init__()
        if backbone == "resnet34":
            resnet = models.resnet34(
                weights=models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
            )
            feat_dim = 512
        elif backbone == "resnet18":
            resnet = models.resnet18(
                weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            )
            feat_dim = 512
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # 去掉最后的分类层
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # [B, feat_dim, 1, 1]
        self.fc = nn.Linear(feat_dim, 7)  # 输出 tx, ty, tz, qw, qx, qy, qz

    def forward(self, x):
        f = self.backbone(x).flatten(1)   # [B, feat_dim]
        y = self.fc(f)                    # [B, 7]
        t, q = y[:, :3], y[:, 3:]
        q = q / (q.norm(dim=1, keepdim=True) + 1e-8)  # 归一化 quaternion
        return torch.cat([t, q], dim=1)
