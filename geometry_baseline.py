import cv2
import numpy as np
from utils import tq_to_pose


class GeometryBaseline:
    def __init__(self, intrinsics, method="ORB"):
        """
        ORB/SIFT + PnP + RANSAC baseline
        Args:
            intrinsics: 相机内参 [fx, fy, cx, cy]
            method: "ORB" or "SIFT"
        """
        self.fx, self.fy, self.cx, self.cy = intrinsics

        if method == "ORB":
            self.detector = cv2.ORB_create(5000)
        elif method == "SIFT":
            self.detector = cv2.SIFT_create()
        else:
            raise ValueError("method must be ORB or SIFT")

        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING if method == "ORB" else cv2.NORM_L2, crossCheck=True)

    def estimate_pose(self, img1, depth1, img2, depth2, pose2_gt=None):
        """
        估计相机姿态 (img1 对齐到 img2 的位姿)
        Args:
            img1: RGB 图 (query)
            depth1: 深度图 (对应 img1)
            img2: RGB 图 (reference)
            depth2: 深度图 (对应 img2) [可选]
            pose2_gt: 如果提供，可以用作参考
        Returns:
            pose: [4,4] 相机外参矩阵
        """
        # --- 1. 特征提取 ---
        kps1, des1 = self.detector.detectAndCompute(cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY), None)
        kps2, des2 = self.detector.detectAndCompute(cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY), None)

        if des1 is None or des2 is None:
            print("No features detected!")
            return None

        # --- 2. 特征匹配 ---
        matches = self.matcher.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)[:500]  # 取前 500 个匹配点

        pts2D, pts3D = [], []
        for m in matches:
            u, v = kps1[m.queryIdx].pt  # img1 的像素坐标
            u, v = int(u), int(v)

            # 深度
            if depth1[v, u] == 0:
                continue
            Z = depth1[v, u] / 1000.0  # 深度 (mm → m)

            X = (u - self.cx) * Z / self.fx
            Y = (v - self.cy) * Z / self.fy

            pts3D.append([X, Y, Z])
            pts2D.append(kps2[m.trainIdx].pt)

        pts3D, pts2D = np.array(pts3D, dtype=np.float32), np.array(pts2D, dtype=np.float32)

        if len(pts3D) < 6:
            print("Not enough 3D-2D matches!")
            return None

        # --- 3. PnP + RANSAC ---
        K = np.array([[self.fx, 0, self.cx],
                      [0, self.fy, self.cy],
                      [0, 0, 1]], dtype=np.float32)

        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            pts3D, pts2D, K, None,
            flags=cv2.SOLVEPNP_ITERATIVE,
            reprojectionError=3.0,
            iterationsCount=100
        )

        if not success:
            print("PnP failed!")
            return None

        R, _ = cv2.Rodrigues(rvec)
        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = R
        pose[:3, 3] = tvec.squeeze()

        return pose


# ---------------- Demo ----------------
if __name__ == "__main__":
    # 假设 Kinect 相机内参 (7-Scenes 默认参数)
    fx, fy, cx, cy = 585.0, 585.0, 320.0, 240.0
    baseline = GeometryBaseline([fx, fy, cx, cy], method="ORB")

    # Demo: 用两帧图像估计相对位姿 (需要你加载 RGB+depth)
    img1 = cv2.imread("frame-000000.color.png")
    depth1 = cv2.imread("frame-000000.depth.png", -1)
    img2 = cv2.imread("frame-000100.color.png")
    depth2 = cv2.imread("frame-000100.depth.png", -1)

    pose = baseline.estimate_pose(img1, depth1, img2, depth2)
    print("Estimated pose:\n", pose)
