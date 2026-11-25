"""DermaMNIST 7 分类：小 CNN + 数据增强 + 类别加权 + TTA + 多随机种子平均。

用法（在解压好的 DermaMNIST 文件夹旁边）：
    python train.py --data DermaMNIST_work/DermaMNIST --out out
产物：
    out/submission.csv   test 集预测（直接可交）
    out/val_pred.csv     验证集预测（可用 score.py 自查）
"""
import argparse
import csv
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
MEAN = np.array([0.7635, 0.5461, 0.5705], dtype=np.float32)
STD = np.array([0.1409, 0.1523, 0.1695], dtype=np.float32)


def load_split(folder: Path):
    """读一个 split，返回 (图像数组 N,H,W,3 uint8, 标签数组 或 None, id 列表)。"""
    df = pd.read_csv(folder / "labels.csv")
    imgs = np.stack([
        np.asarray(Image.open(folder / fn).convert("RGB"), dtype=np.uint8)
        for fn in df["filename"]
    ])
    ids = [Path(fn).stem for fn in df["filename"]]
    return imgs, df["label"].to_numpy(np.int64), ids


def load_test(folder: Path):
    files = sorted(folder.glob("*.png"))
    imgs = np.stack([np.asarray(Image.open(f).convert("RGB"), dtype=np.uint8) for f in files])
    return imgs, [f.stem for f in files]


def to_tensor(imgs_u8):
    x = imgs_u8.astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    return torch.from_numpy(x.transpose(0, 3, 1, 2))


def augment(x, gen):
    """随机翻转 + 随机旋转/缩放/平移 + 亮度对比度抖动，全部向量化。"""
    n = x.shape[0]
    hf = torch.rand(n, 1, 1, 1, generator=gen) < 0.5
    vf = torch.rand(n, 1, 1, 1, generator=gen) < 0.5
    x = torch.where(hf, torch.flip(x, dims=[3]), x)
    x = torch.where(vf, torch.flip(x, dims=[2]), x)

    # 仿射变换：旋转 ±15°，缩放 0.9~1.1，平移 ±12%
    ang = (torch.rand(n, generator=gen) * 2 - 1) * (15 * np.pi / 180)
    scale = 1 + (torch.rand(n, generator=gen) * 2 - 1) * 0.1
    tx = (torch.rand(n, generator=gen) * 2 - 1) * 0.12
    ty = (torch.rand(n, generator=gen) * 2 - 1) * 0.12
    cos, sin = torch.cos(ang) * scale, torch.sin(ang) * scale
    theta = torch.zeros(n, 2, 3)
    theta[:, 0, 0], theta[:, 0, 1], theta[:, 0, 2] = cos, -sin, tx
    theta[:, 1, 0], theta[:, 1, 1], theta[:, 1, 2] = sin, cos, ty
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    x = F.grid_sample(x, grid, padding_mode="reflection", align_corners=False)

    bright = 1 + (torch.rand(n, 1, 1, 1, generator=gen) - 0.5) * 0.4
    x = x * bright
    contrast = 1 + (torch.rand(n, 1, 1, 1, generator=gen) - 0.5) * 0.4
    m = x.mean(dim=(1, 2, 3), keepdim=True)
    x = (x - m) * contrast + m
    return x


class SmallCNN(nn.Module):
    def __init__(self, num_classes=7, width=32):
        super().__init__()

        def block(cin, cout, n_conv):
            layers = []
            for i in range(n_conv):
                layers += [nn.Conv2d(cin if i == 0 else cout, cout, 3, padding=1, bias=False),
                           nn.BatchNorm2d(cout), nn.ReLU(inplace=True)]
            layers.append(nn.MaxPool2d(2))
            return nn.Sequential(*layers)

        self.features = nn.Sequential(
            block(3, width, 2),        # 28 -> 14
            block(width, width * 2, 2),  # 14 -> 7
            block(width * 2, width * 4, 2),  # 7 -> 3
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(0.3), nn.Linear(width * 4, num_classes),
        )

    def forward(self, x):
        return self.head(self.features(x))


def tta_probs(model, x, batch=256):
    """原图 + 水平翻转 + 垂直翻转 + 双向翻转，四份概率取平均。"""
    model.eval()
    out = torch.zeros(x.shape[0], 7)
    with torch.no_grad():
        for i in range(0, x.shape[0], batch):
            xb = x[i:i + batch]
            views = [xb, torch.flip(xb, dims=[3]), torch.flip(xb, dims=[2]),
                     torch.flip(xb, dims=[2, 3])]
            p = sum(torch.softmax(model(v), dim=1) for v in views) / len(views)
            out[i:i + batch] = p
    return out.numpy()


def balanced_accuracy(y_true, y_pred):
    recalls = []
    for k in range(7):
        m = y_true == k
        if m.sum():
            recalls.append((y_pred[m] == k).mean())
    return float(np.mean(recalls))


def macro_f1(y_true, y_pred):
    f1s = []
    for k in range(7):
        tp = float(((y_true == k) & (y_pred == k)).sum())
        n_true = float((y_true == k).sum())
        n_pred = float((y_pred == k).sum())
        if n_true and (n_true + n_pred):
            f1s.append(2 * tp / (n_true + n_pred))
    return float(np.mean(f1s))


def train_one(seed, x_train, y_train, x_val, y_val, x_test, epochs, width, lr, bs, device):
    gen = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    model = SmallCNN(width=width).to(device)

    counts = np.bincount(y_train, minlength=7).astype(np.float32)
    w = (counts.sum() / (7 * counts)) ** 0.5  # 平衡但不过分偏向极少数类
    w = torch.tensor(w / w.mean(), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=w, label_smoothing=0.05)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps_per_epoch = int(np.ceil(len(x_train) / bs))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, epochs=epochs,
                                                steps_per_epoch=steps_per_epoch)

    best_bacc, best_state = -1.0, None
    for ep in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        perm = torch.randperm(len(x_train), generator=gen)
        total_loss = 0.0
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]
            xb = augment(x_train[idx], gen).to(device)
            yb = y_train[idx].to(device)
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            opt.step()
            sched.step()
            total_loss += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            vp = np.concatenate([torch.argmax(model(x_val[i:i + 512].to(device)), dim=1).cpu().numpy()
                                 for i in range(0, len(x_val), 512)])
        bacc = balanced_accuracy(y_val.numpy(), vp)
        acc = float((vp == y_val.numpy()).mean())
        flag = ""
        if bacc > best_bacc:
            best_bacc = bacc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            flag = " *"
        print(f"  seed {seed} epoch {ep:02d}/{epochs} loss {total_loss / len(x_train):.4f} "
              f"val_acc {acc:.4f} val_bacc {bacc:.4f} ({time.time() - t0:.1f}s){flag}", flush=True)

    model.load_state_dict(best_state)
    return model, best_bacc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="DermaMNIST_work/DermaMNIST")
    ap.add_argument("--out", default="derma/out")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--seeds", default="1")
    args = ap.parse_args()

    torch.set_num_threads(os.cpu_count() or 4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备：{device}，线程：{torch.get_num_threads()}")

    data = Path(args.data)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    x_tr_u8, y_tr, _ = load_split(data / "train")
    x_va_u8, y_va, val_ids = load_split(data / "val")
    x_te_u8, test_ids = load_test(data / "test")
    print(f"train {len(x_tr_u8)}  val {len(x_va_u8)}  test {len(x_te_u8)}")
    print("训练集类别分布：", dict(zip(CLASSES, np.bincount(y_tr, minlength=7).tolist())))

    x_train, y_train = to_tensor(x_tr_u8), torch.from_numpy(y_tr)
    x_val, y_val = to_tensor(x_va_u8), torch.from_numpy(y_va)
    x_test = to_tensor(x_te_u8)

    seeds = [int(s) for s in str(args.seeds).split(",") if s.strip()]
    val_prob_sum, test_prob_sum = 0, 0
    for s in seeds:
        model, bacc = train_one(s, x_train, y_train, x_val, y_val, x_test,
                                args.epochs, args.width, args.lr, args.bs, device)
        val_prob_sum = val_prob_sum + tta_probs(model, x_val)
        test_prob_sum = test_prob_sum + tta_probs(model, x_test)
    val_prob = val_prob_sum / len(seeds)
    test_prob = test_prob_sum / len(seeds)

    val_pred = val_prob.argmax(1)
    print(f"\n[验证集] accuracy {float((val_pred == y_va).mean()):.4f}  "
          f"balanced accuracy {balanced_accuracy(y_va, val_pred):.4f}  "
          f"macro-F1 {macro_f1(y_va, val_pred):.4f}")

    pd.DataFrame({"id": val_ids, "label": val_pred}).to_csv(out / "val_pred.csv", index=False)
    test_pred = test_prob.argmax(1)
    pd.DataFrame({"id": test_ids, "label": test_pred}).to_csv(out / "submission.csv", index=False)
    np.save(out / "test_prob.npy", test_prob)
    np.save(out / "val_prob.npy", val_prob)
    print(f"已写出 {out / 'submission.csv'}（{len(test_pred)} 行）和 {out / 'val_pred.csv'}")
    print("测试集预测分布：", dict(zip(CLASSES, np.bincount(test_pred, minlength=7).tolist())))


if __name__ == "__main__":
    main()
