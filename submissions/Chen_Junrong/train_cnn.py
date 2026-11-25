#!/usr/bin/env python3
"""DermaMNIST 七分类基线：小 CNN，CPU 可跑，不需要 GPU，也不需要 torchvision。

用法：
    python train_cnn.py                 # 训练 + 验证 + 预测 test -> submission.csv
    python train_cnn.py --epochs 16
    python train_cnn.py --predict-only  # 只加载已有权重做预测

口径：val 上的 accuracy / balanced accuracy / macro-F1 / 每类 recall 直接用数据包里
score.py 的 score()、macro_f1()，和官方打分口径一致。数据不平衡（黑素细胞痣占 67%），
所以训练用类别权重，并按 balanced accuracy 选最佳模型。
"""
import argparse
import csv
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

HERE = Path(__file__).resolve().parent
WORK = HERE / "work"
K = 7


# ---------- 找数据 ----------
def find_data_dir(start: Path) -> Path:
    for p in [start, *start.parents]:
        cand = p / "DermaMNIST"
        if (cand / "train" / "labels.csv").exists():
            return cand
    raise SystemExit("找不到 DermaMNIST/ 目录（需要 train/labels.csv）")


def load_labels(split_dir: Path, csv_name="labels.csv"):
    out = {}
    with open(split_dir / csv_name, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            fn = r["filename"]
            out[fn[:-4] if fn.lower().endswith(".png") else fn] = int(r["label"])
    return out


def load_images(split_dir: Path, ids):
    x = np.zeros((len(ids), 28, 28, 3), dtype=np.uint8)
    for i, k in enumerate(ids):
        x[i] = np.asarray(Image.open(split_dir / (k + ".png")).convert("RGB"), dtype=np.uint8)
    return x


def to_tensor(x_u8):
    """uint8 (N,28,28,3) -> float (N,3,28,28)，缩放到 [-1,1]"""
    t = torch.from_numpy(x_u8).float().div(127.5).sub(1.0)
    return t.permute(0, 3, 1, 2).contiguous()


# ---------- 增强 ----------
def augment(xb, pad=2):
    b = xb.shape[0]
    m = torch.rand(b) < 0.5
    if m.any():
        xb[m] = torch.flip(xb[m], dims=[3])
    m = torch.rand(b) < 0.5
    if m.any():
        xb[m] = torch.flip(xb[m], dims=[2])
    k = int(torch.randint(0, 4, (1,)))
    if k:
        xb = torch.rot90(xb, k, dims=[2, 3])
    xp = F.pad(xb, (pad, pad, pad, pad), mode="reflect")
    offs = torch.randint(0, 2 * pad + 1, (b, 2))
    return torch.stack([xp[i, :, offs[i, 0]:offs[i, 0] + 28, offs[i, 1]:offs[i, 1] + 28] for i in range(b)])


# ---------- 模型 ----------
class Net(nn.Module):
    """28x28 输入的小 CNN，约 40 万参数。"""

    def __init__(self, n_cls=K, w=32, p=0.2):
        super().__init__()

        def blk(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False),
                                 nn.BatchNorm2d(o), nn.ReLU(inplace=True))

        self.body = nn.Sequential(
            blk(3, w), blk(w, w), nn.MaxPool2d(2),            # 14x14
            blk(w, 2 * w), blk(2 * w, 2 * w), nn.MaxPool2d(2),  # 7x7
            blk(2 * w, 4 * w), nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(p), nn.Linear(4 * w, n_cls))

    def forward(self, x):
        return self.head(self.body(x))


# ---------- 官方口径指标 ----------
def load_score_module(data_dir: Path):
    spec = importlib.util.spec_from_file_location("official_score", data_dir / "score.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def metrics_from_probs(probs: torch.Tensor, ids, y, smod):
    pred = probs.argmax(1).tolist()
    preds = {i: int(p) for i, p in zip(ids, pred)}
    answers = {i: int(v) for i, v in zip(ids, y.tolist())}
    bacc, acc, recalls, cm = smod.score(preds, answers)
    return acc, bacc, smod.macro_f1(cm), recalls, cm


@torch.no_grad()
def tta_probs(model, x, bs=512):
    """八重/四重 TTA：恒等 + 左右翻 + 上下翻 + 旋转 180，概率平均。"""
    outs = []
    for fn in (lambda t: t, lambda t: torch.flip(t, dims=[3]),
               lambda t: torch.flip(t, dims=[2]), lambda t: torch.rot90(t, 2, dims=[2, 3])):
        chunks = [F.softmax(model(fn(x[i:i + bs])), dim=1) for i in range(0, len(x), bs)]
        outs.append(torch.cat(chunks))
    return torch.stack(outs).mean(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=14)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--save-probs", action="store_true", help="把 val/test 的概率存到 work/，供 ensemble 平均")
    ap.add_argument("--predict-only", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    WORK.mkdir(exist_ok=True)

    data_dir = find_data_dir(HERE)
    smod = load_score_module(data_dir)
    tr_dir, va_dir, te_dir = data_dir / "train", data_dir / "val", data_dir / "test"

    tr_lab, va_lab = load_labels(tr_dir), load_labels(va_dir)
    tr_ids, va_ids = sorted(tr_lab), sorted(va_lab)
    x_tr, y_tr = to_tensor(load_images(tr_dir, tr_ids)), torch.tensor([tr_lab[k] for k in tr_ids])
    x_va, y_va = to_tensor(load_images(va_dir, va_ids)), torch.tensor([va_lab[k] for k in va_ids])
    print(f"train {tuple(x_tr.shape)}  val {tuple(x_va.shape)}")

    ckpt = WORK / "model.pt"

    if not args.predict_only:
        # 类别权重：按频次开方倒数，压住 67% 的黑素细胞痣，但不至于把多数类压死
        cnt = np.bincount(y_tr.numpy(), minlength=K).astype(np.float64)
        w = (cnt.sum() / cnt) ** 0.5
        w = torch.tensor(w / w.mean(), dtype=torch.float32)
        print("类别权重", dict(zip(range(K), np.round(w.numpy(), 2))))

        model = Net(w=args.width)
        print("参数量 %.0f 万" % (sum(p.numel() for p in model.parameters()) / 1e4))
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
        steps = args.epochs * ((len(x_tr) + args.bs - 1) // args.bs)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.25)
        crit = nn.CrossEntropyLoss(weight=w, label_smoothing=0.05)

        best = (-1.0, -1, None)  # (bal_acc, epoch, state)
        for ep in range(1, args.epochs + 1):
            t0 = time.time()
            model.train()
            perm = torch.randperm(len(x_tr))
            tot = corr = 0
            run = 0.0
            for i in range(0, len(perm), args.bs):
                idx = perm[i:i + args.bs]
                xb, yb = augment(x_tr[idx].clone()), y_tr[idx]
                logit = model(xb)
                loss = crit(logit, yb)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                sched.step()
                run += loss.item() * len(idx)
                corr += (logit.argmax(1) == yb).sum().item()
                tot += len(idx)
            model.eval()
            probs = tta_probs(model, x_va)
            acc, bacc, f1, rec, _ = metrics_from_probs(probs, va_ids, y_va, smod)
            flag = ""
            if bacc > best[0]:
                best = (bacc, ep, {k: v.detach().clone() for k, v in model.state_dict().items()})
                flag = "  *best*"
            print(f"ep{ep:02d} loss {run/tot:.3f} tracc {corr/tot:.3f} | "
                  f"val acc {acc:.4f} bacc {bacc:.4f} F1 {f1:.4f} | {time.time()-t0:.1f}s{flag}")

        model.load_state_dict(best[2])
        torch.save({"state": best[2], "width": args.width, "epoch": best[1]}, ckpt)

    # ---------- 用最佳权重评估 + 预测 ----------
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    model = Net(w=blob["width"])
    model.load_state_dict(blob["state"])
    model.eval()
    print(f"\n用第 {blob['epoch']} 轮的最佳权重")

    probs = tta_probs(model, x_va)
    acc, bacc, f1, rec, cm = metrics_from_probs(probs, va_ids, y_va, smod)
    print(f"val  accuracy {acc:.4f}  balanced accuracy {bacc:.4f}  macro-F1 {f1:.4f}")
    print("每类 recall: " + "  ".join(f"{k}:{r:.3f}" for k, r in enumerate(rec)))
    print("混淆矩阵（行=真实，列=预测）:")
    for k in range(K):
        print("  " + " ".join(f"{v:5d}" for v in cm[k]))

    # val 预测存一份，方便用官方脚本再核一次
    with open(WORK / "val_pred.csv", "w", encoding="utf-8", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["id", "label"])
        for i, p in zip(va_ids, probs.argmax(1).tolist()):
            wr.writerow([i, p])

    # test 预测
    te_ids = sorted(p.stem for p in te_dir.glob("*.png"))
    x_te = to_tensor(load_images(te_dir, te_ids))
    tp = tta_probs(model, x_te).argmax(1).tolist()
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["id", "label"])
        for i, p in zip(te_ids, tp):
            wr.writerow([i, p])
    dist = np.bincount(tp, minlength=K)
    print(f"\n写出 {args.out}：{len(te_ids)} 行")
    print("预测分布 " + "  ".join(f"{k}:{v}({v/len(tp)*100:.1f}%)" for k, v in enumerate(dist)))

    if args.save_probs:
        np.save(WORK / f"val_probs_seed{args.seed}.npy", probs.numpy())
        np.save(WORK / f"test_probs_seed{args.seed}.npy", tta_probs(model, x_te).numpy())
        (WORK / f"ids_seed{args.seed}.json").write_text(
            json.dumps({"val": va_ids, "test": te_ids}, ensure_ascii=False), encoding="utf-8")
        print(f"已存概率：work/*_seed{args.seed}.npy（供 ensemble 平均）")
    (WORK / "metrics.json").write_text(json.dumps(
        {"acc": acc, "bacc": bacc, "f1": f1, "recalls": rec, "epoch": blob["epoch"],
         "val_ids": len(va_ids)}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
