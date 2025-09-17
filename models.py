import torch
import torch.nn as nn
import torchvision.models as models


class PoseNet(nn.Module):
    def __init__(self, backbone="resnet18", pretrained=True, feat_dim=2048):
        """
        PoseNet implementation with ResNet backbone
        Args:
            backbone: "resnet18", "resnet34", "resnet50"
            pretrained: 是否加载 ImageNet 预训练权重
            feat_dim: 特征维度 (resnet18/34=512, resnet50=2048)
        """
        super(PoseNet, self).__init__()

        # ---- backbone ----
        if backbone == "resnet18":
            net = models.resnet18(pretrained=pretrained)
            feat_dim = 512
        elif backbone == "resnet34":
            net = models.resnet34(pretrained=pretrained)
            feat_dim = 512
        elif backbone == "resnet50":
            net = models.resnet50(pretrained=pretrained)
            feat_dim = 2048
        else:
            raise ValueError("Unsupported backbone")

        # 去掉 ResNet 最后一层分类器 (fc)
        self.feature_extractor = nn.Sequential(*list(net.children())[:-1])

        # ---- Pose 回归层 ----
        self.fc_pose = nn.Linear(feat_dim, 7)

    def forward(self, x):
        """
        Args:
            x: [B, 3, 224, 224] 图像
        Returns:
            pose: [B, 7] (t[3], q[4])
        """
        feat = self.feature_extractor(x)  # [B, feat_dim, 1, 1]
        feat = feat.view(feat.size(0), -1)  # [B, feat_dim]
        pose = self.fc_pose(feat)           # [B, 7]
        return pose


# ---------------- Demo ----------------
if __name__ == "__main__":
    model = PoseNet(backbone="resnet18", pretrained=False)
    x = torch.randn(2, 3, 224, 224)  # batch=2
    out = model(x)
    print("Input shape:", x.shape)
    print("Output shape:", out.shape)  # [2, 7]
