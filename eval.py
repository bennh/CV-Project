import torch
from torch.utils.data import DataLoader
import numpy as np
from data_loader import SevenScenesDataset
from utils import pose_error
import torchvision.transforms as T


def evaluate(model, dataset, device="cuda"):
    """
    在整个测试集上评估模型
    Args:
        model: 已训练的 PoseNet 模型
        dataset: SevenScenesDataset (split="test")
        device: "cuda" or "cpu"
    Returns:
        mean_t_err: 平均平移误差 (m)
        mean_r_err: 平均旋转误差 (deg)
    """
    model.eval()
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    t_errors, r_errors = [], []

    with torch.no_grad():
        for batch in loader:
            img = batch["image"].to(device)
            pose_gt = batch["pose_matrix"].squeeze(0).cpu().numpy()

            # ---- 模型预测 (假设输出 t[3], q[4]) ----
            pred = model(img)  # e.g. (B,7)
            pred = pred.squeeze(0).cpu().numpy()

            t_pred = pred[:3]
            q_pred = pred[3:] / np.linalg.norm(pred[3:])  # 归一化

            # 预测的 pose matrix
            from utils import tq_to_pose
            pose_pred = tq_to_pose(t_pred, q_pred)

            # 误差
            t_err, r_err = pose_error(pose_gt, pose_pred)
            t_errors.append(t_err)
            r_errors.append(r_err)

    mean_t_err = np.mean(t_errors)
    mean_r_err = np.mean(r_errors)

    print(f"Evaluation Results:")
    print(f"  Mean Translation Error: {mean_t_err:.3f} m")
    print(f"  Mean Rotation Error:    {mean_r_err:.3f} °")

    return mean_t_err, mean_r_err


if __name__ == "__main__":
    # 数据预处理 (与训练保持一致)
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])

    # 加载测试集
    dataset = SevenScenesDataset(root_dir="7-scenes-dataset",
                                 scene="chess", split="test",
                                 transform=transform)

    # 模型 (示例：PoseNet)
    # TODO: 替换为你训练好的模型
    from models import PoseNet
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PoseNet().to(device)

    # 加载权重
    # model.load_state_dict(torch.load("checkpoints/posenet_chess.pth"))

    # 评估
    evaluate(model, dataset, device)
