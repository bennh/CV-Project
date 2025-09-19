# Deep Learning for Camera Pose Estimation: A Comparative Study on 7Scenes

## Overview

This project implements and compares **learning-based** and **geometry-based** methods for camera pose estimation on the **7-Scenes dataset**:

- **PoseNet (CNN regression)**: trained on one scene, tested on the same and generalized to other scenes.
- **PnP + RANSAC baseline**: classic geometric approach using 2D–3D correspondences.

---

## Project Structure

```text
project/
│── data\_loader.py          # 7-Scenes dataset loader
│── models.py               # PoseNet model definition
│── train.py                # Train PoseNet
│── eval.py                 # Evaluate PoseNet (generalization experiments)
│── geometry\_baseline.py    # PnP + RANSAC baseline
│── utils.py                # Helper functions (pose errors, quaternion ops, I/O)
│── demo.ipynb              # Final notebook: experiments + plots
│── README.md               # Documentation
````

---

## Environment Setup

Recommended: Python 3.10 + PyTorch

```bash
conda create -n pose python=3.10 -y
conda activate pose

pip install torch torchvision opencv-python tqdm matplotlib pandas scikit-image faiss-cpu
````

---

## Dataset: 7-Scenes

Download 7-Scenes dataset and organize as (https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/):

```
7Scenes/
  chess/
    TrainSplit.txt
    TestSplit.txt
    sequence1/
      frame-000000.color.png
      frame-000000.depth.png
      frame-000000.pose.txt
      ...
    ...
  pumpkin/
  redkitchen/
```

* `TrainSplit.txt` and `TestSplit.txt` specify which sequences are used for training/testing.

---

## How to Run

### 1. Train PoseNet

Train on **chess** scene:

```bash
python train.py \
  --data_root /path/to/7Scenes \
  --scene chess \
  --epochs 30 \
  --batch_size 32 \
  --lr 1e-4 \
  --beta 120.0 \
  --out runs/posenet_chess
```

This will save:

* `best.ckpt` (model weights)
* `loss_curve.png` and `loss_curve.csv`

---

### 2. Evaluate PoseNet

Evaluate on **same scene** and **generalization scenes**:

```bash
python eval.py \
  --data_root /path/to/7Scenes \
  --scene chess \
  --ckpt runs/posenet_chess/best.ckpt \
  --out results/chess_eval.csv

python eval.py --data_root /path/to/7Scenes --scene pumpkin --ckpt runs/posenet_chess/best.ckpt --out results/pumpkin_eval.csv
python eval.py --data_root /path/to/7Scenes --scene redkitchen --ckpt runs/posenet_chess/best.ckpt --out results/redkitchen_eval.csv
```

---

### 3. Run PnP + RANSAC Baseline

```bash
python geometry_baseline.py --data_root /path/to/7Scenes --scene chess --topk 5
python geometry_baseline.py --data_root /path/to/7Scenes --scene pumpkin --topk 5
python geometry_baseline.py --data_root /path/to/7Scenes --scene redkitchen --topk 5
```

---

### 4. Final Notebook

Run `demo.ipynb` to:

* Plot PoseNet training curve
* Evaluate PoseNet on multiple scenes
* Run PnP+RANSAC baseline
* Summarize results in tables & plots

---

## Expected Outputs

* **Training curve**: PoseNet loss vs. epochs
* **Quantitative results**: mean/median translation (m) & rotation (°) errors
* **Comparison**: PoseNet vs. PnP+RANSAC
* **Plots**: error distributions, visualizations

---

## Report

Use results in `demo.ipynb` to write a 6–8 page report:

* Abstract, Introduction
* Related Work
* Method (PoseNet vs. PnP+RANSAC)
* Experiments (with plots & tables)
* Ablation (optional)
* Conclusion

---

## Author

Project for Computer Vision Final Project (SS25)
Heidelberg University
By Binheng Zheng and Yuefeiyang Li

---
