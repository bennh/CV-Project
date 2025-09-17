import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as T

from data_loader import SevenScenesDataset
from models import PoseNet
from utils import pose_error


# ---------- 损失函数 ----------
class PoseLoss(nn.Module):
    def __init__(self, beta=120.0):
        """
        PoseNet 损失函数
        L = ||t - t_gt||_2 + beta * ||q - q_gt||_2
        """
        super(PoseLoss, self).__init__()
        self.beta = beta

    def forward(self, pred, target_t, target_q):
        pred_t = pred[:, :3]
        pred_q = pred[:, 3:]
        pred_q = pred_q / torch.norm(pred_q, dim=1, keepdim=True)  # 归一化

        # L2 平移损失
        t_loss = torch.norm(pred_t - target_t, dim=1).mean()

        # L2 旋转损失
        q_loss = torch.norm(pred_q - target_q, dim=1).mean()

        return t_loss + self.beta * q_loss


# ---------- 训练函数 ----------
def train(model, train_loader, val_loader, device, epochs=20, lr=1e-4, beta=120.0):
    criterion = PoseLoss(beta=beta)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")

    for epoch in range(epochs):
        # ---- Train ----
        model.train()
        train_loss = 0
        for batch in train_loader:
            imgs = batch["image"].to(device)
            t_gt = batch["translation"].to(device)
            q_gt = batch["quaternion"].to(device)

            optimizer.zero_grad()
            preds = model(imgs)

            loss = criterion(preds, t_gt, q_gt)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # ---- Validation ----
        model.eval()
        val_loss = 0
        t_errs, r_errs = [], []
        with torch.no_grad():
            for batch in val_loader:
                imgs = batch["image"].to(device)
                t_gt = batch["translation"].cpu().numpy()
                q_gt = batch["quaternion"].cpu().numpy()
                pose_gt = batch["pose_matrix"].cpu().numpy()

                preds = model(imgs).cpu().numpy()
                t_pred, q_pred = preds[0, :3], preds[0, 3:]
                q_pred /= np.linalg.norm(q_pred)

                # 损失
                loss = criterion(torch.tensor(preds), batch["translation"], batch["quaternion"])
                val_loss += loss.item()

                # 误差 (只用 batch=1 时比较合理)
                from utils import tq_to_pose
                pose_pred = tq_to_pose(t_pred, q_pred)
                t_err, r_err = pose_error(pose_gt[0], pose_pred)
                t_errs.append(t_err)
                r_errs.append(r_err)

        val_loss /= len(val_loader)
        mean_t_err = sum(t_errs) / len(t_errs)
        mean_r_err = sum(r_errs) / len(r_errs)

        print(f"[Epoch {epoch+1}/{epochs}] "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"T_err: {mean_t_err:.3f} m | R_err: {mean_r_err:.2f}°")

        # ---- 保存最好模型 ----
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"posenet_best.pth")
            print("  ✅ Saved best model.")


# ---------- Main ----------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])

    # 加载数据集 (这里用 chess 场景举例)
    train_set = SevenScenesDataset("7-scenes-dataset", scene="chess", split="train", transform=transform)
    val_set   = SevenScenesDataset("7-scenes-dataset", scene="chess", split="test", transform=transform)

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_set, batch_size=1, shuffle=False)

    # 定义模型
    model = PoseNet(backbone="resnet18", pretrained=True).to(device)

    # 训练
    train(model, train_loader, val_loader, device, epochs=20, lr=1e-4, beta=120.0)
