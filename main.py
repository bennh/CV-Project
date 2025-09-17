import torch
import numpy as np
import torchvision.transforms as T
from torch.utils.data import DataLoader

from data_loader import SevenScenesDataset
from models import PoseNet
from utils import pose_error, tq_to_pose
from geometry_baseline import GeometryBaseline
from train import train  # 直接调用已有的训练函数


def evaluate_posenet(model, dataset, device="cuda"):
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    t_errs, r_errs = [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            img = batch["image"].to(device)
            pose_gt = batch["pose_matrix"].squeeze(0).cpu().numpy()

            pred = model(img).squeeze(0).cpu().numpy()
            t_pred, q_pred = pred[:3], pred[3:] / np.linalg.norm(pred[3:])
            pose_pred = tq_to_pose(t_pred, q_pred)

            t_err, r_err = pose_error(pose_gt, pose_pred)
            t_errs.append(t_err)
            r_errs.append(r_err)
    return np.mean(t_errs), np.mean(r_errs)


def evaluate_pnp(dataset, intrinsics):
    baseline = GeometryBaseline(intrinsics, method="ORB")
    t_errs, r_errs = [], []
    for i in range(len(dataset)-1):
        sample = dataset[i]
        sample_next = dataset[i+1]

        img1 = sample["image"].permute(1,2,0).numpy() * 255
        img1 = img1.astype(np.uint8)
        depth1 = sample["depth"].numpy()

        img2 = sample_next["image"].permute(1,2,0).numpy() * 255
        img2 = img2.astype(np.uint8)
        depth2 = sample_next["depth"].numpy()

        pose_gt = sample["pose_matrix"].numpy()

        pose_pred = baseline.estimate_pose(img1, depth1, img2, depth2)
        if pose_pred is None:
            continue

        t_err, r_err = pose_error(pose_gt, pose_pred)
        t_errs.append(t_err)
        r_errs.append(r_err)

    if len(t_errs) == 0:
        return np.nan, np.nan
    return np.mean(t_errs), np.mean(r_errs)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])

    # 相机内参 (7-Scenes Kinect 默认值)
    fx, fy, cx, cy = 585.0, 585.0, 320.0, 240.0
    intrinsics = [fx, fy, cx, cy]

    # -------- 1. 在 chess 场景训练 PoseNet --------
    print("\n=== Training PoseNet on chess ===")
    train_set = SevenScenesDataset("7-scenes-dataset", scene="chess", split="train", transform=transform)
    val_set   = SevenScenesDataset("7-scenes-dataset", scene="chess", split="test", transform=transform)
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_set, batch_size=1, shuffle=False)

    model = PoseNet(backbone="resnet18", pretrained=True).to(device)
    train(model, train_loader, val_loader, device, epochs=10, lr=1e-4, beta=120.0)  # 少训练点节省时间
    model.load_state_dict(torch.load("posenet_best.pth", map_location=device))

    # -------- 2. 在 chess, fire, office 上评估 --------
    scenes = ["chess", "fire", "office"]
    results = []

    for scene in scenes:
        print(f"\n=== Evaluating on {scene} ===")

        # PoseNet
        test_set_posenet = SevenScenesDataset("7-scenes-dataset", scene=scene, split="test", transform=transform)
        t_err_pn, r_err_pn = evaluate_posenet(model, test_set_posenet, device)

        # PnP+RANSAC
        test_set_pnp = SevenScenesDataset("7-scenes-dataset", scene=scene, split="test", transform=transform, return_depth=True)
        t_err_pnp, r_err_pnp = evaluate_pnp(test_set_pnp, intrinsics)

        results.append([scene, t_err_pn, r_err_pn, t_err_pnp, r_err_pnp])

    # -------- 3. 输出结果表 --------
    print("\n=== Final Results (3 scenes) ===")
    print(f"{'Scene':<12} {'PoseNet_T(m)':<12} {'PoseNet_R(°)':<12} {'PnP_T(m)':<12} {'PnP_R(°)':<12}")
    for row in results:
        print(f"{row[0]:<12} {row[1]:<12.3f} {row[2]:<12.2f} {row[3]:<12.3f} {row[4]:<12.2f}")
