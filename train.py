# train.py

import os
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pandas as pd

from data_loader import SevenScenesDataset
from models import PoseNet
from utils import PoseLoss, set_seed, save_ckpt


def train(args):
    set_seed(0)

    # Dataset
    train_ds = SevenScenesDataset(args.data_root, args.scene, split="train")
    train_dl = DataLoader(train_ds, batch_size=args.batch_size,
                          shuffle=True, num_workers=4, pin_memory=True)

    # Model
    model = PoseNet('resnet34')
    criterion = PoseLoss(beta=args.beta)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # Save directory
    os.makedirs(args.out, exist_ok=True)

    # Training
    best_loss = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        for img, t_gt, q_gt in train_dl:

            pred = model(img)
            loss = criterion(pred, t_gt, q_gt)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * img.size(0)

        avg_loss = total_loss / len(train_ds)
        history.append({"epoch": epoch, "loss": avg_loss})
        print(f"[{epoch}/{args.epochs}] avg_loss = {avg_loss:.4f}")

        # Save best checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_ckpt(model, args.out, "best.ckpt")

    # Save loss curve
    df = pd.DataFrame(history)
    df.to_csv(os.path.join(args.out, "loss_curve.csv"), index=False)

    plt.figure()
    plt.plot(df["epoch"], df["loss"], marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title(f"PoseNet Training ({args.scene})")
    plt.grid(True)
    plt.savefig(os.path.join(args.out, "loss_curve.png"))
    plt.close()

    print(f"Training finished. Best loss = {best_loss:.4f}")
    print(f"Results saved in {args.out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True,
                        help="Root directory of 7Scenes dataset")
    parser.add_argument("--scene", type=str, default="chess",
                        help="Which scene to train on (e.g., chess)")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--beta", type=float, default=120.0,
                        help="Weight factor between translation and rotation loss")
    parser.add_argument("--out", type=str, required=True,
                        help="Output directory for checkpoints and logs")
    args = parser.parse_args()

    train(args)
