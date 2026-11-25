#!/usr/bin/env python3
"""把 train_cnn.py --save-probs 存下的多个模型概率平均，输出 submission.csv。

用法：
    python average.py work                # 全部模型等权平均（默认，最稳）
    python average.py work --select       # 枚举所有组合，只在 val 上比一比，用于排查
    python average.py work --alpha 0.2    # 额外做类别先验修正（对 balanced accuracy 友好，
                                          # 但 accuracy 和 macro-F1 会掉，见 README 里的取舍表）
    python average.py work 1 3            # 只用 seed 1 和 3

评估口径直接用数据包里 score.py 的 score()/macro_f1()，和官方一致。
"""
import glob
import itertools
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

import train_cnn as T


def load_seeds(work: Path):
    out = {}
    for f in sorted(glob.glob(str(work / "val_probs_seed*.npy"))):
        s = int(re.search(r"seed(\d+)", f).group(1))
        t, i = work / f"test_probs_seed{s}.npy", work / f"ids_seed{s}.json"
        if t.exists() and i.exists():
            out[s] = (np.load(f), np.load(t), json.loads(i.read_text(encoding="utf-8")))
    return out


def apply_prior(p, priors, alpha):
    if not alpha:
        return p
    q = p / (priors ** alpha)
    return q / q.sum(1, keepdims=True)


def main():
    argv = sys.argv[1:]
    select = "--select" in argv
    argv = [a for a in argv if a != "--select"]
    alpha = 0.0
    if "--alpha" in argv:
        j = argv.index("--alpha")
        alpha = float(argv[j + 1])
        del argv[j:j + 2]
    work = Path(argv[0] if argv else "work")
    only = [int(a) for a in argv[1:]]

    seeds = load_seeds(work)
    if only:
        seeds = {k: v for k, v in seeds.items() if k in only}
    if not seeds:
        raise SystemExit(f"{work} 下没找到 *_seed*.npy，先跑 train_cnn.py --save-probs")

    data_dir = T.find_data_dir(T.HERE)
    smod = T.load_score_module(data_dir)
    va_lab = T.load_labels(data_dir / "val")
    va_ids = sorted(va_lab)
    y = torch.tensor([va_lab[k] for k in va_ids])
    for k, (_, _, ids) in seeds.items():
        if ids["val"] != va_ids:
            raise SystemExit(f"seed{k} 的 val id 顺序和数据不一致")
    priors = np.bincount(list(T.load_labels(data_dir / "train").values()), minlength=7).astype(float)
    priors /= priors.sum()

    keys = sorted(seeds)
    print(f"可用模型 {keys}" + (f"，先验修正 alpha={alpha}" if alpha else ""))
    for k in keys:
        pv = apply_prior(seeds[k][0], priors, alpha)
        acc, bacc, f1, _, _ = T.metrics_from_probs(torch.tensor(pv), va_ids, y, smod)
        print(f"  单个 seed{k}: acc {acc:.4f}  bacc {bacc:.4f}  F1 {f1:.4f}")

    if select:
        best = (-1.0, None)
        for r in range(1, len(keys) + 1):
            for combo in itertools.combinations(keys, r):
                pv = apply_prior(np.mean([seeds[k][0] for k in combo], axis=0), priors, alpha)
                res = T.metrics_from_probs(torch.tensor(pv), va_ids, y, smod)
                print(f"  {combo}: acc {res[0]:.4f}  bacc {res[1]:.4f}  F1 {res[2]:.4f}")
                if res[2] > best[0]:
                    best = (res[2], combo)
        print(f"  （仅参考）val 上 F1 最高的组合 {best[1]}，F1 {best[0]:.4f}")
        print("  注：1003 张 val 上 0.01 级别的差距是噪声，多组合里挑最高会过拟合 val，"
              "所以默认仍然全模型平均。")

    combo = tuple(keys)
    pv = apply_prior(np.mean([seeds[k][0] for k in combo], axis=0), priors, alpha)
    acc, bacc, f1, rec, cm = T.metrics_from_probs(torch.tensor(pv), va_ids, y, smod)
    print(f"\n最终用 {combo} 平均：acc {acc:.4f}  bacc {bacc:.4f}  macro-F1 {f1:.4f}")
    print("每类 recall: " + "  ".join(f"{k}:{r:.3f}" for k, r in enumerate(rec)))
    print("混淆矩阵（行=真实，列=预测）:")
    for k in range(7):
        print("  " + " ".join(f"{v:5d}" for v in cm[k]))

    # val 预测存一份，方便用官方脚本再核一次
    with open(work / "val_pred.csv", "w", encoding="utf-8", newline="") as f:
        f.write("id,label\n")
        f.writelines(f"{i},{int(p)}\n" for i, p in zip(va_ids, pv.argmax(1).tolist()))

    test_ids = seeds[combo[0]][2]["test"]
    tp = apply_prior(np.mean([seeds[k][1] for k in combo], axis=0), priors, alpha)
    pred = tp.argmax(1)
    out = T.HERE / "submission.csv"
    with open(out, "w", encoding="utf-8", newline="") as f:
        f.write("id,label\n")
        f.writelines(f"{i},{int(p)}\n" for i, p in zip(test_ids, pred))
    dist = np.bincount(pred, minlength=7)
    print(f"\n写出 {out}：{len(test_ids)} 行")
    print("预测分布 " + "  ".join(f"{k}:{v}({v/len(pred)*100:.1f}%)" for k, v in enumerate(dist)))
    (work / "metrics.json").write_text(json.dumps(
        {"seeds": list(combo), "alpha": alpha, "acc": acc, "bacc": bacc, "f1": f1, "recalls": rec},
        ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
