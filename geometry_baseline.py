# geometry_baseline.py

import os
import argparse
import glob
import numpy as np
import cv2
import torch
import torchvision.models as models
import torchvision.transforms as T
import pandas as pd

from data_loader import load_pose_file
from utils import pose_err_trans_m, pose_err_angular_deg, se3_to_tq


# ======================
# 辅助函数
# ======================
def global_descriptor(img_bgr, backbone, transform):
    """提取全局描述子 (ResNet18 GAP)。"""
    with torch.no_grad():
        x = transform(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)).unsqueeze(0)
        f = backbone(x).flatten(1).numpy()[0]
        f /= (np.linalg.norm(f) + 1e-8)
    return f


def backproject_keypoint(px, depth, K):
    """像素 + 深度 → 相机坐标系 3D 点。"""
    u, v = int(round(px[0])), int(round(px[1]))
    if u < 0 or v < 0 or u >= depth.shape[1] or v >= depth.shape[0]:
        return None
    z = depth[v, u]
    if z <= 0:
        return None
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return np.array([x, y, z], dtype=np.float32)


def cam_to_world(Xc, T_w_c):
    """相机坐标系点 → 世界坐标系。"""
    R, t = T_w_c[:3, :3], T_w_c[:3, 3:4]
    return (R @ Xc.reshape(3, 1) + t).reshape(3)


def solve_pnp_ransac(pts2d, pts3d, K):
    """PnP + RANSAC 求解相机位姿。"""
    if len(pts3d) < 6:
        return None
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        np.array(pts3d, np.float32),
        np.array(pts2d, np.float32),
        K, None,
        iterationsCount=2000,
        reprojectionError=3.0,
        confidence=0.999
    )
    if not ok:
        return None
    R, _ = cv2.Rodrigues(rvec)
    T_w_c = np.eye(4, dtype=np.float32)
    T_w_c[:3, :3] = R
    T_w_c[:3, 3] = tvec.reshape(-1)
    return T_w_c, inliers


# ======================
# 主函数
# ======================
def run_baseline(args):
    scene_dir = os.path.join(args.data_root, args.scene)

    # 相机内参
    intr_path = os.path.join(scene_dir, "intrinsics.txt")
    fx, fy, cx, cy = np.loadtxt(intr_path).astype(np.float32).tolist()
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

    # ORB + BFMatcher
    orb = cv2.ORB_create(nfeatures=args.orb_kpts)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    # 全局特征提取网络 (ResNet18 GAP)
    resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    backbone = torch.nn.Sequential(*list(resnet.children())[:-1]).eval()
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((256, 320)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225])
    ])

    # 构建 train DB
    print("Building train DB descriptors...")
    train_split = os.path.join(scene_dir, "TrainSplit.txt")
    with open(train_split) as f:
        train_seqs = [line.strip() for line in f]

    train_rgb, train_depth, train_pose = [], [], []
    db_feats = []

    for seq in train_seqs:
        seq_dir = os.path.join(scene_dir, seq)
        rgb_files = sorted(glob.glob(os.path.join(seq_dir, "*.color.png")))
        for rgb_path in rgb_files:
            base = rgb_path.replace(".color.png", "")
            depth_path = base + ".depth.png"
            pose_path = base + ".pose.txt"
            if not os.path.exists(depth_path) or not os.path.exists(pose_path):
                continue
            img = cv2.imread(rgb_path)
            f = global_descriptor(img, backbone, transform)
            db_feats.append(f)
            train_rgb.append(rgb_path)
            train_depth.append(depth_path)
            train_pose.append(pose_path)

    db_feats = np.array(db_feats)

    # 遍历 test
    test_split = os.path.join(scene_dir, "TestSplit.txt")
    with open(test_split) as f:
        test_seqs = [line.strip() for line in f]

    results = []
    frame_id = 0

    for seq in test_seqs:
        seq_dir = os.path.join(scene_dir, seq)
        rgb_files = sorted(glob.glob(os.path.join(seq_dir, "*.color.png")))

        for rgb_path in rgb_files:
            base = rgb_path.replace(".color.png", "")
            pose_path = base + ".pose.txt"
            query_img = cv2.imread(rgb_path)

            # query 全局特征
            q_feat = global_descriptor(query_img, backbone, transform)

            # 相似度检索 top-k
            sims = db_feats @ q_feat
            idxs = np.argsort(sims)[-args.topk:][::-1]

            query_kp, query_desc = orb.detectAndCompute(query_img, None)
            if query_desc is None:
                frame_id += 1
                continue

            best = None
            for j in idxs:
                ref_img = cv2.imread(train_rgb[j])
                ref_kp, ref_desc = orb.detectAndCompute(ref_img, None)
                if ref_desc is None:
                    continue

                matches = bf.knnMatch(query_desc, ref_desc, k=2)
                good = [m for m, n in matches if m.distance < 0.75 * n.distance]
                if len(good) < 20:
                    continue

                depth_ref = cv2.imread(train_depth[j], cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
                T_ref, _, _ = load_pose_file(train_pose[j])

                pts2d, pts3d = [], []
                for m in good:
                    qpt = query_kp[m.queryIdx].pt
                    rpt = ref_kp[m.trainIdx].pt
                    Xc = backproject_keypoint(rpt, depth_ref, K)
                    if Xc is None:
                        continue
                    Xw = cam_to_world(Xc, T_ref)
                    pts2d.append(qpt)
                    pts3d.append(Xw)

                res = solve_pnp_ransac(pts2d, pts3d, K)
                if res is None:
                    continue
                T_est, inliers = res
                if best is None or len(inliers) > len(best[2]):
                    best = (T_est, pts2d, inliers)

            if best is None:
                frame_id += 1
                continue

            T_est = best[0]
            T_gt, t_gt, q_gt = load_pose_file(pose_path)
            t_est, q_est = se3_to_tq(T_est)

            t_err = pose_err_trans_m(t_est, t_gt)
            r_err = pose_err_angular_deg(q_est, q_gt)

            results.append({
                "frame": frame_id,
                "t_err_m": t_err,
                "r_err_deg": r_err,
                "t_gt_x": float(t_gt[0]), "t_gt_y": float(t_gt[1]), "t_gt_z": float(t_gt[2]),
                "t_pred_x": float(t_est[0]), "t_pred_y": float(t_est[1]), "t_pred_z": float(t_est[2]),
            })
            frame_id += 1

    # 转 DataFrame
    df = pd.DataFrame(results)

    stats = {
        "scene": args.scene,
        "mean_t": df["t_err_m"].mean(),
        "median_t": df["t_err_m"].median(),
        "mean_r": df["r_err_deg"].mean(),
        "median_r": df["r_err_deg"].median()
    }

    print(f"[PnP+RANSAC | {args.scene}] "
          f"mean_t={stats['mean_t']:.3f} m | median_t={stats['median_t']:.3f} m || "
          f"mean_r={stats['mean_r']:.2f}° | median_r={stats['median_r']:.2f}°")

    # 保存
    os.makedirs("results", exist_ok=True)
    per_frame_csv = os.path.join("results", f"{args.scene}_baseline_perframe.csv")
    df.to_csv(per_frame_csv, index=False)

    stats_csv = os.path.join("results", f"{args.scene}_baseline.csv")
    pd.DataFrame([stats]).to_csv(stats_csv, index=False)

    return stats, df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--scene", type=str, required=True,
                        help="Scene to evaluate (e.g., chess, pumpkin, redkitchen)")
    parser.add_argument("--orb_kpts", type=int, default=2000)
    parser.add_argument("--topk", type=int, default=5,
                        help="Number of retrieved nearest neighbors")
    args = parser.parse_args()

    run_baseline(args)
