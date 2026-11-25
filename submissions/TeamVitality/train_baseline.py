"""DermaMNIST 基线 v1：颜色直方图 + 7x7 缩略图特征，类别过采样后训练小 MLP。

用法：  python train_baseline.py        # 训练并在 val 上评估，生成 submission.csv
"""

import csv
import time
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

ROOT = Path(__file__).resolve().parent.parent / "DermaMNIST"


def load_split(split):
    folder = ROOT / split
    names = sorted(p.name for p in folder.glob("*.png"))
    x = np.stack([
        np.asarray(Image.open(folder / n).convert("RGB"), dtype=np.float32).ravel() / 255.0
        for n in names
    ])
    return names, x


def load_labels(split, names):
    table = {}
    with open(ROOT / split / "labels.csv", newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            table[row["filename"]] = int(row["label"])
    return np.array([table[n] for n in names])


def featurize(flat_img, bins=16, small=7):
    """每张图 -> 48 维三通道颜色直方图 + 147 维 7x7 缩略图，共 195 维。"""
    img = flat_img.reshape(1, 28, 28, 3)
    step = 28 // small
    hists = np.concatenate([
        np.histogram(img[:, :, :, c], bins=bins, range=(0, 1), density=True)[0]
        for c in range(3)
    ])
    thumb = img.reshape(-1, small, step, small, step, 3).mean(axis=(2, 4)).ravel()
    return np.concatenate([hists, thumb])


def build_model():
    return make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(512, 256),
            alpha=1e-3,
            batch_size=128,
            max_iter=400,
            early_stopping=True,
            n_iter_no_change=15,
            random_state=0,
        ),
    )


def main():
    t0 = time.time()
    tr_names, x_tr = load_split("train")
    y_tr = load_labels("train", tr_names)
    va_names, x_va = load_split("val")
    y_va = load_labels("val", va_names)
    te_names, x_te = load_split("test")

    f_tr = np.stack([featurize(r) for r in x_tr])
    f_va = np.stack([featurize(r) for r in x_va])
    f_te = np.stack([featurize(r) for r in x_te])

    # 少数类过采样到与最大类同量（让模型不只押最多数那一类）
    top = np.bincount(y_tr).max()
    idx = np.concatenate([
        resample(np.where(y_tr == c)[0], replace=True, n_samples=top, random_state=0)
        for c in range(7)
    ])

    model = build_model()
    model.fit(f_tr[idx], y_tr[idx])
    pred_va = model.predict(f_va)
    print(f"val accuracy          {(pred_va == y_va).mean():.4f}")
    print(f"val balanced accuracy {balanced_accuracy_score(y_va, pred_va):.4f}")
    print(f"val macro-F1          {f1_score(y_va, pred_va, average='macro'):.4f}")

    pred_te = model.predict(f_te)
    out = ROOT / "submission.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "label"])
        for n, p in zip(te_names, pred_te):
            w.writerow([n[:-4], int(p)])
    print(f"wrote {out}  ({len(pred_te)} rows, {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
