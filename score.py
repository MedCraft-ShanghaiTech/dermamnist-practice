"""DermaMNIST 实践题打分脚本。只依赖 Python 标准库，本文件里没有任何答案。

用法：
  打分：      python score.py 预测.csv 答案.csv
              组织者：答案用 answers.csv（测试集，不发给学生；不写第二个参数时默认用本脚本同目录下的 answers.csv）
              学生：  答案用 DermaMNIST/val/labels.csv，给自己的验证集预测打分（id 写成 val_00000 这样）
  只查格式：  python score.py --check submission.csv DermaMNIST/sample_submission.csv
              只核对表头、行数、id、类别编号，不打分。

批量打分脚本 make_leaderboard.py 会 import 本文件的 read_answers / read_submission / score。
"""
import csv
import sys
from pathlib import Path

NAMES = ["光化性角化病和上皮内癌", "基底细胞癌", "良性角化样病变", "皮肤纤维瘤",
         "黑色素瘤", "黑素细胞痣", "血管病变"]
K = len(NAMES)


class 格式错误(Exception):
    pass


def read_answers(path):
    """读答案文件。id 列可以叫 id（answers.csv、sample_submission.csv），
    也可以叫 filename（val/labels.csv，会去掉 .png）。"""
    with open(path, encoding="utf-8-sig", newline="") as f:
        out = {}
        for r in csv.DictReader(f):
            rid = r["id"] if "id" in r else r["filename"]
            if rid.endswith(".png"):
                rid = rid[:-4]
            out[rid] = int(r["label"])
        return out


def read_submission(path, answer_ids):
    p = Path(path)
    if not p.exists():
        raise 格式错误(f"找不到文件：{p}")
    try:
        text = p.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        raise 格式错误("文件不是 UTF-8 编码。请用 Python（pandas.to_csv）保存，不要用 Excel 另存。")
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        raise 格式错误("文件是空的。")
    rows = list(csv.reader(lines))
    header = [h.strip().lower() for h in rows[0]]
    if "id" not in header or "label" not in header:
        raise 格式错误(f"第一行表头必须包含 id 和 label 两列，现在是：{rows[0]}。"
                    "（常见原因：保存时没写表头，或者列名写成了别的，比如 image / pred）")
    i_id, i_lab = header.index("id"), header.index("label")
    body = rows[1:]
    errors, preds, warned_float = [], {}, False
    for n, r in enumerate(body, start=2):
        if len(errors) >= 10:
            break
        if len(r) <= max(i_id, i_lab):
            errors.append(f"第 {n} 行列数不够：{r}")
            continue
        rid, lab = r[i_id].strip(), r[i_lab].strip()
        if rid.endswith(".png"):
            errors.append(f"第 {n} 行 id 是 '{rid}'：id 要去掉 .png 后缀，写成 '{rid[:-4]}'")
            continue
        try:
            v = float(lab)
        except ValueError:
            errors.append(f"第 {n} 行 label 是 '{lab}'：label 必须是 0–6 的整数编号，不能是类名")
            continue
        if v != int(v) or not (0 <= int(v) < K):
            errors.append(f"第 {n} 行 label 是 '{lab}'：类别编号只能是 0、1、2、3、4、5、6")
            continue
        if "." in lab:
            warned_float = True
        if rid in preds:
            errors.append(f"第 {n} 行 id '{rid}' 重复出现")
            continue
        preds[rid] = int(v)
    if errors:
        raise 格式错误("发现以下问题（最多列 10 条）：\n  " + "\n  ".join(errors))
    unknown = [i for i in preds if i not in answer_ids]
    missing = [i for i in answer_ids if i not in preds]
    if unknown:
        example = next(iter(answer_ids))
        raise 格式错误(f"有 {len(unknown)} 个 id 不在答案里，例如 {unknown[:3]}。"
                    f"id 应该是图片文件名去掉 .png，比如 {example}。")
    if missing:
        raise 格式错误(f"应有 {len(answer_ids)} 行预测，现在只有 {len(preds)} 行，"
                    f"缺 {len(missing)} 个 id，例如 {missing[:3]}。")
    if warned_float:
        print("提醒：label 写成了小数形式（如 5.0），已按整数处理；下次请存成整数。\n")
    return preds


def score(preds, answers):
    cm = [[0] * K for _ in range(K)]
    for i, y in answers.items():
        cm[y][preds[i]] += 1
    recalls = [cm[k][k] / sum(cm[k]) for k in range(K)]
    acc = sum(cm[k][k] for k in range(K)) / len(answers)
    return sum(recalls) / K, acc, recalls, cm


def main():
    args = sys.argv[1:]
    check_only = "--check" in args
    args = [a for a in args if a != "--check"]
    if not args:
        print(__doc__)
        sys.exit(1)
    ans_path = Path(args[1]) if len(args) > 1 else Path(__file__).with_name("answers.csv")
    if not ans_path.exists():
        print(f"找不到答案文件：{ans_path}\n" + __doc__)
        sys.exit(1)
    answers = read_answers(ans_path)
    try:
        preds = read_submission(args[0], answers.keys())
    except 格式错误 as e:
        print("提交格式有误，未打分。\n" + str(e))
        sys.exit(2)
    if check_only:
        print(f"格式没问题：表头、{len(preds)} 行、id、类别编号都对得上。")
        return
    bacc, acc, recalls, cm = score(preds, answers)
    print(f"平衡准确率 balanced accuracy：{bacc:.4f}")
    print(f"准确率 accuracy：            {acc:.4f}")
    print("\n每类 recall（这一类的图里猜对的比例）：")
    for k in range(K):
        print(f"  {k} {NAMES[k]:<12}{recalls[k]:.3f}  （{cm[k][k]}/{sum(cm[k])}）")
    print("\n混淆矩阵（行 = 真实类别，列 = 预测类别）：")
    print("      " + "".join(f"{j:>6}" for j in range(K)))
    for k in range(K):
        print(f"  {k}   " + "".join(f"{v:>6}" for v in cm[k]))


if __name__ == "__main__":
    main()
