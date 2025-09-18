# geometry_baseline.py

import os
import argparse
import glob
import numpy as np
import cv2
import torch
from data_loader import SevenScenesDataset, load_pose_file
from utils import pose_err_trans_m, pose_err_angular_deg, se3_to_tq


def backproject_keypoint(px, depth, K):
    """将像素坐标 + 深度投影到相机坐标系 (Xc)。"""
    u, v = int(round(px[0])), int(round(px[1]))
    if u < 0 or v < 0 or u >= depth.shape[1] or v >= depth.shape[0]:
        return None
    z = depth[v, u]
    if z <= 0:  # 无效深度
        return None
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return np.array([x, y, z], dtype=np.float32)


def cam_to_world(Xc, T_w_c):
    """相机坐标系点转到世界坐标系。"""
    R = T_w_c[:3, :3]
    t = T_w_c[:3, 3:4]
    Xw = (R @ Xc.reshape(3, 1) + t).reshape(3)
    return Xw


def solve_pnp_ransac(pts2d, pts3d, K):
    """用 PnP + RANSAC 估计相机姿态。"""
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


def run_baseline(args):
    scene_dir = os.path.join(args.data_root, args.scene)

    # 载入内参
    intr_path = os.path.join(scene_dir, "intrinsics.txt")
    fx, fy, cx, cy = np.loadtxt(intr_path).astype(np.float32).tolist()
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

    # ORB
    orb = cv2.ORB_create(nfeatures=args.orb_kpts)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # 遍历 test 图像
    test_split = os.path.join(scene_dir, "TestSplit.txt")
    with open(test_split) as f:
        test_seqs = [line.strip() for line in f]

    results_t, results_r = [], []

    for seq in test_seqs:
        seq_dir = os.path.join(scene_dir, seq)
        rgb_files = sorted(glob.glob(os.path.join(seq_dir, "*.color.png")))

        for rgb_path in rgb_files:
            base = rgb_path.replace(".color.png", "")
            depth_path = base + ".depth.png"
            pose_path = base + ".pose.txt"

            # 加载 query 图像
            query_img = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
            query_kp, query_desc = orb.detectAndCompute(query_img, None)
            if query_desc is None:
                continue

            # 选一个 train 集合中的参考帧（简单随机 / 可改为检索）
            train_split = os.path.join(scene_dir, "TrainSplit.txt")
            with open(train_split) as f:
                train_seqs = [line.strip() for line in f]

            ref_seq = np.random.choice(train_seqs)
            ref_dir = os.path.join(scene_dir, ref_seq)
            ref_rgb = sorted(glob.glob(os.path.join(ref_dir, "*.color.png")))
            if not ref_rgb:
                continue
            ref_path = np.random.choice(ref_rgb)
            ref_depth = ref_path.replace(".color.png", ".depth.png")
            ref_pose = ref_path.replace(".color.png", ".pose.txt")

            ref_img = cv2.imread(ref_path, cv2.IMREAD_COLOR)
            ref_kp, ref_desc = orb.detectAndCompute(ref_img, None)
            if ref_desc is None:
                continue

            # 匹配
            matches = bf.match(query_desc, ref_desc)
            matches = sorted(matches, key=lambda x: x.distance)[:args.max_matches]

            depth_ref = cv2.imread(ref_depth, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
            T_ref, _, _ = load_pose_file(ref_pose)

            pts2d, pts3d = [], []
            for m in matches:
                q_pt = query_kp[m.queryIdx].pt
                r_pt = ref_kp[m.trainIdx].pt
                Xc = backproject_keypoint(r_pt, depth_ref, K)
                if Xc is None:
                    continue
                Xw = cam_to_world(Xc, T_ref)
                pts2d.append(q_pt)
                pts3d.append(Xw)

            if len(pts3d) < 6:
                continue

            res = solve_pnp_ransac(pts2d, pts3d, K)
            if res is None:
                continue
            T_est, inliers = res

            # GT pose
            T_gt, t_gt, q_gt = load_pose_file(pose_path)
            t_est, q_est = se3_to_tq(T_est)

            t_err = pose_err_trans_m(t_est, t_gt)
            r_err = pose_err_angular_deg(q_est, q_gt)
            results_t.append(t_err)
            results_r.append(r_err)

    # 统计
    results_t = np.array(results_t)
    results_r = np.array(results_r)

    print(f"[PnP+RANSAC | {args.scene}] "
          f"mean_t = {results_t.mean():.3f} m | median_t = {np.median(results_t):.3f} m || "
          f"mean_r = {results_r.mean():.2f}° | median_r = {np.median(results_r):.2f}°")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True,
                        help="Root directory of 7Scenes dataset")
    parser.add_argument("--scene", type=str, required=True,
                        help="Scene to evaluate (e.g., chess, pumpkin, redkitchen)")
    parser.add_argument("--orb_kpts", type=int, default=2000,
                        help="Number of ORB keypoints")
    parser.add_argument("--max_matches", type=int, default=500,
                        help="Max number of matches to use for PnP")
    args = parser.parse_args()

    run_baseline(args)
