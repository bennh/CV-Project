# geometry_baseline.py
# 单序列内评估的 PnP+RANSAC 基线：
# - 对 TestSplit.txt 中的每个 seq-XX，均在该序列内部做检索+PnP
# - 避免了 7-Scenes 各序列世界坐标不一致的问题
# - 输出：逐帧误差、每序列统计、场景级统计

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
# 工具函数
# ======================
def global_descriptor(img_bgr, backbone, transform):
    """提取全局描述子 (ResNet18 GAP)。CPU-only。"""
    with torch.no_grad():
        x = transform(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)).unsqueeze(0)
        f = backbone(x).flatten(1).numpy()[0]
        f /= (np.linalg.norm(f) + 1e-8)
    return f


def backproject_keypoint(px, depth, K):
    """像素 + 深度 → 相机坐标系 3D 点（单位：米）。"""
    u, v = int(round(px[0])), int(round(px[1]))
    if u < 0 or v < 0 or u >= depth.shape[1] or v >= depth.shape[0]:
        return None
    z = float(depth[v, u])
    if z <= 0:
        return None
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return np.array([x, y, z], dtype=np.float32)


def cam_to_world(Xc, T_w_c):
    """相机系点 → 世界系点，使用相机到世界的 4x4 变换 T_w_c。"""
    R, t = T_w_c[:3, :3], T_w_c[:3, 3:4]
    return (R @ Xc.reshape(3, 1) + t).reshape(3)


def solve_pnp_ransac(pts2d, pts3d, K, reproj_err=5.0, iterations=2000, conf=0.999):
    """PnP + RANSAC，返回 T_w_c（相机到世界）与 inliers。"""
    if len(pts3d) < 6:
        return None
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        np.asarray(pts3d, np.float32),
        np.asarray(pts2d, np.float32),
        K, None,
        iterationsCount=int(iterations),
        reprojectionError=float(reproj_err),
        confidence=float(conf)
    )
    if not ok:
        return None

    # OpenCV 返回世界->相机 (R_cw, t_cw)，需要取逆得到相机->世界
    R_cw, _ = cv2.Rodrigues(rvec)
    t_cw = tvec.reshape(3, 1)
    R_wc = R_cw.T
    t_wc = -R_wc @ t_cw

    T_w_c = np.eye(4, dtype=np.float32)
    T_w_c[:3, :3] = R_wc
    T_w_c[:3, 3] = t_wc.ravel()
    return T_w_c, inliers


# ======================
# 主流程：单序列内评估 + 场景级汇总
# ======================
def run_baseline(args):
    """
    对 args.scene 的 TestSplit.txt 中列出的每个 seq-XX：
      - 在该序列内部构建候选库并进行检索 + ORB 匹配 + PnP/RANSAC
      - 计算逐帧误差
    最后把所有序列汇总，得到场景级 mean/median。
    """
    scene_dir = os.path.join(args.data_root, args.scene)

    # 相机内参：若没有 intrinsics.txt，使用 Kinect 常用默认参数
    intr_path = os.path.join(scene_dir, "intrinsics.txt")
    if os.path.exists(intr_path):
        fx, fy, cx, cy = np.loadtxt(intr_path).astype(np.float32).tolist()
    else:
        print(f"[Warn] {intr_path} not found. Using default intrinsics 585/585/320/240.")
        fx, fy, cx, cy = 585.0, 585.0, 320.0, 240.0
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

    # 局部/全局特征
    orb = cv2.ORB_create(nfeatures=args.orb_kpts)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    backbone = torch.nn.Sequential(*list(resnet.children())[:-1]).eval()
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((256, 320)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225])
    ])

    # 读取将要评估的序列（仅用 TestSplit）
    test_split = os.path.join(scene_dir, "TestSplit.txt")
    with open(test_split) as f:
        test_seqs = [line.strip() for line in f if line.strip()]

    all_rows = []
    per_seq_stats = []

    frame_counter = 0

    for seq in test_seqs:
        seq_dir = os.path.join(scene_dir, seq)
        rgb_files = sorted(glob.glob(os.path.join(seq_dir, "*.color.png")))
        if len(rgb_files) == 0:
            print(f"[Warn] No frames found in {seq_dir}. Skipping.")
            continue

        # === 构建该序列内部的检索库 ===
        print(f"[{args.scene}/{seq}] building per-sequence DB ...")
        feats, depth_files, pose_files = [], [], []
        for rgb_path in rgb_files:
            base = rgb_path.replace(".color.png", "")
            depth_path = base + ".depth.png"
            pose_path = base + ".pose.txt"
            if not (os.path.exists(depth_path) and os.path.exists(pose_path)):
                continue
            img = cv2.imread(rgb_path)
            feats.append(global_descriptor(img, backbone, transform))
            depth_files.append(depth_path)
            pose_files.append(pose_path)
        feats = np.asarray(feats)
        if len(feats) != len(rgb_files):
            # 对齐：如果有缺文件的帧，简单地过滤使三者长度一致
            keep = [i for i,(r,d,p) in enumerate(zip(rgb_files, depth_files, pose_files))
                    if os.path.exists(d) and os.path.exists(p)]
            rgb_files = [rgb_files[i] for i in keep]
            feats = feats[[i for i in range(len(keep))]]
            depth_files = [depth_files[i] for i in range(len(keep))]
            pose_files = [pose_files[i] for i in range(len(keep))]

        # === 遍历该序列内的 query 帧 ===
        seq_rows = []
        for qi, q_rgb in enumerate(rgb_files):
            q_img = cv2.imread(q_rgb)
            q_feat = global_descriptor(q_img, backbone, transform)

            # 在本序列内检索（排除自身）
            sims = feats @ q_feat
            sims[qi] = -1e9
            idxs = np.argsort(sims)[-args.topk:][::-1]

            q_kp, q_desc = orb.detectAndCompute(q_img, None)
            if q_desc is None:
                frame_counter += 1
                continue

            best = None
            for j in idxs:
                ref_img = cv2.imread(rgb_files[j])
                r_kp, r_desc = orb.detectAndCompute(ref_img, None)
                if r_desc is None:
                    continue

                matches = bf.knnMatch(q_desc, r_desc, k=2)
                good = [m for m, n in matches if m.distance < args.ratio * n.distance]
                if len(good) < args.min_match:
                    continue

                depth_ref = cv2.imread(depth_files[j], cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
                T_ref, _, _ = load_pose_file(pose_files[j])  # 同序列世界系

                pts2d, pts3d = [], []
                for m in good:
                    qpt = q_kp[m.queryIdx].pt
                    rpt = r_kp[m.trainIdx].pt
                    Xc = backproject_keypoint(rpt, depth_ref, K)
                    if Xc is None:
                        continue
                    Xw = cam_to_world(Xc, T_ref)  # 提升到该序列的世界坐标
                    pts2d.append(qpt)
                    pts3d.append(Xw)

                res = solve_pnp_ransac(pts2d, pts3d, K,
                                       reproj_err=args.reproj,
                                       iterations=args.iters,
                                       conf=args.conf)
                if res is None:
                    continue
                T_est, inliers = res
                if best is None or len(inliers) > len(best[2]):
                    best = (T_est, pts2d, inliers)

            if best is None:
                frame_counter += 1
                continue

            T_est = best[0]
            T_gt, t_gt, q_gt = load_pose_file(pose_files[qi])  # 同序列 GT，可直接对齐
            t_est, q_est = se3_to_tq(T_est)

            t_err = pose_err_trans_m(t_est, t_gt)
            r_err = pose_err_angular_deg(q_est, q_gt)

            row = {
                "scene": args.scene,
                "sequence": seq,
                "frame": frame_counter,
                "t_err_m": t_err,
                "r_err_deg": r_err,
                "t_gt_x": float(t_gt[0]), "t_gt_y": float(t_gt[1]), "t_gt_z": float(t_gt[2]),
                "t_pred_x": float(t_est[0]), "t_pred_y": float(t_est[1]), "t_pred_z": float(t_est[2]),
                "inliers": int(len(best[2]))
            }
            seq_rows.append(row)
            all_rows.append(row)
            frame_counter += 1

        # 序列级统计
        if len(seq_rows) > 0:
            df_seq = pd.DataFrame(seq_rows)
            per_seq_stats.append({
                "scene": args.scene,
                "sequence": seq,
                "mean_t": df_seq["t_err_m"].mean(),
                "median_t": df_seq["t_err_m"].median(),
                "mean_r": df_seq["r_err_deg"].mean(),
                "median_r": df_seq["r_err_deg"].median(),
                "mean_inliers": df_seq["inliers"].mean()
            })
            print(f"[{args.scene}|{seq}] "
                  f"mean_t={per_seq_stats[-1]['mean_t']:.3f} m | "
                  f"median_t={per_seq_stats[-1]['median_t']:.3f} m || "
                  f"mean_r={per_seq_stats[-1]['mean_r']:.2f}° | "
                  f"median_r={per_seq_stats[-1]['median_r']:.2f}° | "
                  f"inliers~{per_seq_stats[-1]['mean_inliers']:.1f}")
        else:
            print(f"[{args.scene}|{seq}] no valid estimates.")

    # 汇总到场景级
    if len(all_rows) == 0:
        print(f"[PnP+RANSAC | {args.scene}] no results.")
        return {"scene": args.scene, "mean_t": np.nan, "median_t": np.nan,
                "mean_r": np.nan, "median_r": np.nan}, pd.DataFrame()

    df_all = pd.DataFrame(all_rows)
    scene_stats = {
        "scene": args.scene,
        "mean_t": df_all["t_err_m"].mean(),
        "median_t": df_all["t_err_m"].median(),
        "mean_r": df_all["r_err_deg"].mean(),
        "median_r": df_all["r_err_deg"].median()
    }

    print(f"[PnP+RANSAC | {args.scene}] "
          f"mean_t={scene_stats['mean_t']:.3f} m | median_t={scene_stats['median_t']:.3f} m || "
          f"mean_r={scene_stats['mean_r']:.2f}° | median_r={scene_stats['median_r']:.2f}°")

    # ===== 保存 =====
    os.makedirs("results", exist_ok=True)

    # 逐帧
    per_frame_csv = os.path.join("results", f"{args.scene}_baseline_perframe.csv")
    df_all.to_csv(per_frame_csv, index=False)

    # 序列级统计
    per_seq_csv = os.path.join("results", f"{args.scene}_baseline_perseq.csv")
    pd.DataFrame(per_seq_stats).to_csv(per_seq_csv, index=False)

    # 场景级统计
    stats_csv = os.path.join("results", f"{args.scene}_baseline.csv")
    pd.DataFrame([scene_stats]).to_csv(stats_csv, index=False)

    return scene_stats, df_all


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--scene", type=str, required=True,
                        help="Scene name (e.g., chess, pumpkin, redkitchen)")
    parser.add_argument("--orb_kpts", type=int, default=4000)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--ratio", type=float, default=0.75, help="Lowe's ratio")
    parser.add_argument("--min_match", type=int, default=20, help="min good matches")
    parser.add_argument("--reproj", type=float, default=5.0, help="RANSAC reprojection error")
    parser.add_argument("--iters", type=int, default=2000, help="RANSAC iterations")
    parser.add_argument("--conf", type=float, default=0.999, help="RANSAC confidence")
    args = parser.parse_args()

    run_baseline(args)