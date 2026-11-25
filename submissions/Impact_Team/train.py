#!/usr/bin/env python3
"""DermaMNIST 7 分类：用 train 训练，在 val 上自查，预测 test 生成 submission.csv。

默认从当前目录下的 DermaMNIST 读取数据；输出（submission.csv、val_pred.csv、模型权重）
写到本脚本所在目录（可写）。

关键点：
- ResNet18（ImageNet 预训练）微调，28x28 小图放大到 img_size 训练。
- 类别极度不均衡（5 类占 2/3），用类别加权交叉熵缓解。
- 选 val balanced accuracy 最高的 epoch 作为最终模型。
- test 预测用 TTA（中心裁剪 + 水平翻转 + 两个缩放）平均 logits。
"""
import argparse
import csv
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
K = 7


def read_label_csv(path):
    rows = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            fn = r["filename"]
            if fn.endswith(".png"):
                fn = fn[:-4]
            rows.append((fn, int(r["label"])))
    return rows


class ImageFolderCSV(Dataset):
    """带标签的图片集：samples = [(图片完整路径, label), ...]"""
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


class TestImageDataset(Dataset):
    """无标签测试集：对每张图产出 N 个增强视图（TTA），返回 (N,C,H,W) 张量和下标。"""
    def __init__(self, ids, img_dir, tfms):
        self.ids = ids
        self.img_dir = Path(img_dir)
        self.tfms = tfms

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img = Image.open(self.img_dir / f"{self.ids[idx]}.png").convert("RGB")
        return torch.stack([t(img) for t in self.tfms]), idx


def build_model(num_classes, pretrained=True):
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    m = models.resnet18(weights=weights)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m


def make_tta_transforms(img_size):
    """返回 TTA 用的多个变换：中心裁剪、水平翻转、放大一点裁剪、缩小一点裁剪。"""
    norm = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    base = [transforms.Resize(img_size + 16), transforms.CenterCrop(img_size)]
    flipped = [transforms.Resize(img_size + 16), transforms.CenterCrop(img_size),
               transforms.RandomHorizontalFlip(p=1.0)]
    zoom_in = [transforms.Resize(img_size + 32), transforms.CenterCrop(img_size)]
    zoom_out = [transforms.Resize(img_size), transforms.CenterCrop(img_size)]
    out = []
    for parts in (base, flipped, zoom_in, zoom_out):
        out.append(transforms.Compose(parts + [transforms.ToTensor(), norm]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="DermaMNIST")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent))
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--img", type=int, default=96)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", default="", help="从该 checkpoint 热启动继续训练（不跳过训练）")
    ap.add_argument("--predict_only", action="store_true", help="只加载 --resume 做预测，不训练")
    ap.add_argument("--combine_val", action="store_true", help="把 val 也并入训练集（最终模型用全部标注数据；无 val 自查指标）")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data = Path(args.data)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    train_dir, val_dir, test_dir = data / "train", data / "val", data / "test"

    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")

    train_samples = read_label_csv(train_dir / "labels.csv")
    val_samples = read_label_csv(val_dir / "labels.csv")
    fit_samples = train_samples + (val_samples if args.combine_val else [])
    print(f"train={len(train_samples)}  val={len(val_samples)}  "
          f"fit={len(fit_samples)}{' (train+val)' if args.combine_val else ''}  "
          f"test_imgs={len(list(test_dir.glob('*.png')))}")

    counts = Counter(lab for _, lab in fit_samples)
    print("fit 每类数量:", dict(sorted(counts.items())))
    total = sum(counts.values())
    class_weights = torch.tensor([total / (K * counts[k]) for k in range(K)], dtype=torch.float32).to(device)

    # 用完整路径构造样本列表（combine_val 时 train/val 两目录混合）
    fit_paths = [(train_dir / f"{s}.png", l) for s, l in train_samples]
    if args.combine_val:
        fit_paths += [(val_dir / f"{s}.png", l) for s, l in val_samples]
    val_paths = [(val_dir / f"{s}.png", l) for s, l in val_samples]

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(args.img, scale=(0.6, 1.0), ratio=(0.85, 1.15)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(args.img + 16),
        transforms.CenterCrop(args.img),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    model = build_model(K, pretrained=(not args.resume)).to(device)
    ckpt_path = out / "best_model.pt"
    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location=device))
        print(f"热启动自 {args.resume}")

    if not args.predict_only:
        train_ds = ImageFolderCSV(fit_paths, train_tf)
        train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                                  num_workers=args.workers, drop_last=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)

        if not args.combine_val:
            val_ds = ImageFolderCSV(val_paths, eval_tf)
            val_loader = DataLoader(val_ds, batch_size=args.batch * 2, shuffle=False,
                                    num_workers=args.workers)
            best_bacc = -1.0

        for epoch in range(1, args.epochs + 1):
            model.train()
            tot_loss, n = 0.0, 0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                loss = criterion(model(x), y)
                loss.backward()
                optimizer.step()
                tot_loss += loss.item() * x.size(0)
                n += x.size(0)
            scheduler.step()

            if args.combine_val:
                # val 已并入训练，无独立验证集；保存最终模型
                print(f"epoch {epoch:2d}/{args.epochs}  loss={tot_loss/n:.4f}  (train+val 训练，无验证集)")
                torch.save({k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, ckpt_path)
            else:
                model.eval()
                correct = 0
                cm = np.zeros((K, K), dtype=int)
                with torch.no_grad():
                    for x, y in val_loader:
                        x = x.to(device)
                        pred = model(x).argmax(1).cpu().numpy()
                        y = y.numpy()
                        for t, p in zip(y, pred):
                            cm[t][p] += 1
                        correct += int((pred == y).sum())
                acc = correct / len(val_samples)
                recalls = [cm[k][k] / cm[k].sum() if cm[k].sum() else 0.0 for k in range(K)]
                present = [k for k in range(K) if cm[k].sum() > 0]
                bacc = float(np.mean([recalls[k] for k in present]))
                print(f"epoch {epoch:2d}/{args.epochs}  loss={tot_loss/n:.4f}  val_acc={acc:.4f}  val_bacc={bacc:.4f}")
                if bacc > best_bacc:
                    best_bacc = bacc
                    torch.save({k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, ckpt_path)

        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        if args.combine_val:
            print(f"train+val 训练完成，已加载最终模型 {ckpt_path}")
        else:
            print(f"最优 val_bacc={best_bacc:.4f}，已加载 {ckpt_path}")

    model.eval()

    # val 预测（便于 score.py 自查）；combine_val 时 val 已参与训练，跳过以免指标失真
    if not args.combine_val:
        val_ds = ImageFolderCSV(val_paths, eval_tf)
        val_loader = DataLoader(val_ds, batch_size=args.batch * 2, shuffle=False, num_workers=args.workers)
        val_preds = {}
        with torch.no_grad():
            i = 0
            for x, _ in val_loader:
                x = x.to(device)
                pred = model(x).argmax(1).cpu().numpy()
                for p in pred:
                    val_preds[val_samples[i][0]] = int(p)
                    i += 1
        with open(out / "val_pred.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "label"])
            for stem, _ in val_samples:
                w.writerow([stem, val_preds[stem]])
        print(f"val_pred.csv 写出（{len(val_preds)} 行）")

    # test 预测：严格按 sample_submission.csv 的 id 顺序 + TTA
    sample = data / "sample_submission.csv"
    ids = []
    with open(sample, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            ids.append(r["id"])
    print(f"按 sample_submission.csv 预测 {len(ids)} 张 test 图（TTA）")

    tfms = make_tta_transforms(args.img)
    test_ds = TestImageDataset(ids, test_dir, tfms)
    test_loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False, num_workers=args.workers)
    preds = [0] * len(ids)
    with torch.no_grad():
        for x, idx in test_loader:
            B, N, C, H, W = x.shape
            x = x.view(B * N, C, H, W).to(device)
            logits = model(x).view(B, N, -1).mean(1)
            p = logits.argmax(1).cpu().numpy()
            for j, ii in enumerate(idx.numpy()):
                preds[ii] = int(p[j])

    with open(out / "submission.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "label"])
        for i, lab in zip(ids, preds):
            w.writerow([i, lab])
    print(f"submission.csv 写出（{len(preds)} 行）-> {out / 'submission.csv'}")
    print("校验格式：cd DermaMNIST 目录后运行  python score.py --check <submission.csv 路径>")


if __name__ == "__main__":
    main()
