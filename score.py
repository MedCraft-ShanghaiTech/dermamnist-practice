"""DermaMNIST 实践题：查提交格式、算几项常用指标。只依赖 Python 标准库，本文件里没有任何答案。

用法：
  查格式（交之前跑一次）：
      python score.py --check submission.csv
      不需要答案：对照本文件旁边的 sample_submission.csv（没有就用 test/ 里的图片文件名），
      核对表头、行数、id、类别编号 0–6，不算分。
      sample_submission.csv 不在旁边时，把它的路径写在后面：python score.py --check submission.csv 路径/sample_submission.csv
  在验证集上算指标：
      python score.py val预测.csv val/labels.csv
      val预测.csv 的格式和 submission.csv 一样（两列 id,label），id 写成 val_00000 这样。
      输出 accuracy、balanced accuracy、macro-F1、每类 recall、混淆矩阵。
"""
# 组织者：不写第二个参数时，算分默认用本文件旁边的 answers.csv（测试集答案，不在数据包里）。
# 批量打分脚本 make_leaderboard.py 会 import 本文件的 read_answers / read_submission / score，
# 自查和批量处理共用这一份读文件、算分代码。
import csv
import sys
from pathlib import Path

NAMES = ["光化性角化病和上皮内癌", "基底细胞癌", "良性角化样病变", "皮肤纤维瘤",
         "黑色素瘤", "黑素细胞痣", "血管病变"]
NAMES_EN = ["actinic keratoses and intraepithelial carcinoma", "basal cell carcinoma",
            "benign keratosis-like lesions", "dermatofibroma", "melanoma", "melanocytic nevi", "vascular lesions"]
ABBR = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]   # HAM10000 原始缩写
K = len(NAMES)
NAME_TO_ID = {s.lower(): k for k in range(K) for s in (NAMES[k], NAMES_EN[k], ABBR[k])}
ID_COLS = ["id", "filename", "file_name", "file", "image", "image_id", "img", "name"]
LABEL_COLS = ["label", "pred", "prediction", "predicted_label", "class", "category", "target"]


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


def _clean_id(s):
    s = s.strip().strip('"').replace("\\", "/").rsplit("/", 1)[-1]
    return s[:-4] if s.lower().endswith(".png") else s


def read_submission(path, answer_ids, notes=None):
    """读提交文件，返回 {id: 类别编号}。读不了就抛 格式错误。
    能确定意思的小毛病（表头写成 filename、label 写成类名、多一列行号、id 带 .png、没有表头、
    GBK 编码、分号或 Tab 分隔、label 写成 5.0）照常读出，每条在 notes 里记一句；
    不传 notes 时直接打印"提醒：……"。"""
    answer_ids = set(answer_ids)
    own = []
    p = Path(path)
    if not p.exists():
        raise 格式错误(f"找不到文件：{p}")
    raw = p.read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        try:
            text = raw.decode("gbk")
            own.append("文件是 GBK 编码（多半是 Excel 另存的），已按 GBK 读")
        except UnicodeDecodeError:
            raise 格式错误("文件不是 UTF-8 编码，读不出来。请用 Python（pandas 的 to_csv）保存，不要用 Excel 另存。")
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        raise 格式错误("文件是空的。")
    delim = ","
    if "," not in lines[0]:
        for d, nm in (("\t", "Tab"), (";", "分号")):
            if d in lines[0]:
                delim = d
                own.append(f"列之间用的是{nm}不是逗号，已照读")
                break
    rows = list(csv.reader(lines, delimiter=delim))
    header = [h.strip().strip('"').lower() for h in rows[0]]

    i_id = next((header.index(c) for c in ID_COLS if c in header), None)
    i_lab = next((header.index(c) for c in LABEL_COLS if c in header), None)
    if i_id is not None and i_lab is not None:
        body, start = rows[1:], 2
        if (header[i_id], header[i_lab]) != ("id", "label"):
            own.append(f"表头写成了 {rows[0][i_id].strip()},{rows[0][i_lab].strip()}，已当作 id,label 读")
        extra = [rows[0][j].strip() or "无名列" for j in range(len(header)) if j not in (i_id, i_lab)]
    else:
        # 没认出表头：看第一行是不是已经是数据（某一格去掉 .png 后是测试图的 id）
        hit = [j for j, v in enumerate(rows[0]) if _clean_id(v) in answer_ids]
        if not hit or hit[0] + 1 >= len(rows[0]):
            raise 格式错误(f"第一行表头必须是 id,label 两列，现在是：{rows[0]}。"
                        "（常见原因：保存时没写表头，或者列名写成了别的）")
        i_id, i_lab = hit[0], hit[0] + 1
        body, start = rows, 1
        own.append("没有表头行，已按第一列 id、下一列 label 读")
        extra = [f"第 {j + 1} 列" for j in range(len(rows[0])) if j not in (i_id, i_lab)]
    if extra:
        own.append(f"多了 {len(extra)} 列（{'、'.join(extra[:3])}，多半是 pandas 默认存的行号），已忽略")

    errors, preds = [], {}
    flags = {"png": False, "float": False, "name": False}
    for n, r in enumerate(body, start=start):
        if len(errors) >= 10:
            break
        if len(r) <= max(i_id, i_lab):
            errors.append(f"第 {n} 行列数不够：{r}")
            continue
        rid0, lab = r[i_id].strip(), r[i_lab].strip().strip('"')
        rid = _clean_id(rid0)
        if rid != rid0:
            flags["png"] = True
        if lab.lower() in NAME_TO_ID:
            v = NAME_TO_ID[lab.lower()]
            flags["name"] = True
        else:
            try:
                v = float(lab)
            except ValueError:
                errors.append(f"第 {n} 行 label 是 '{lab}'：label 要填 0–6 的类别编号")
                continue
            if v != int(v) or not (0 <= int(v) < K):
                errors.append(f"第 {n} 行 label 是 '{lab}'：类别编号只能是 0、1、2、3、4、5、6")
                continue
            if "." in lab:
                flags["float"] = True
            v = int(v)
        if rid in preds:
            errors.append(f"第 {n} 行 id '{rid}' 重复出现")
            continue
        preds[rid] = v
    if errors:
        raise 格式错误("发现以下问题（最多列 10 条）：\n  " + "\n  ".join(errors))
    if flags["png"]:
        own.append("id 带了 .png 后缀或文件夹路径，已去掉")
    if flags["name"]:
        own.append("label 写成了类别名字，已按名字换成 0–6 的编号")
    if flags["float"]:
        own.append("label 写成了小数形式（如 5.0），已按整数处理")
    unknown = [i for i in preds if i not in answer_ids]
    missing = [i for i in answer_ids if i not in preds]
    if unknown:
        example = sorted(answer_ids)[0]
        raise 格式错误(f"有 {len(unknown)} 个 id 对不上，例如 {unknown[:3]}。"
                    f"id 应该是图片文件名去掉 .png，比如 {example}。")
    if missing:
        raise 格式错误(f"应有 {len(answer_ids)} 行预测，现在只有 {len(preds)} 行，"
                    f"缺 {len(missing)} 个 id，例如 {sorted(missing)[:3]}。")
    if notes is None:
        for s in own:
            print("提醒：" + s)
        if own:
            print()
    else:
        notes.extend(own)
    return preds


def score(preds, answers):
    """返回 (balanced accuracy, accuracy, 每类 recall, 混淆矩阵)。"""
    cm = [[0] * K for _ in range(K)]
    for i, y in answers.items():
        cm[y][preds[i]] += 1
    recalls = [cm[k][k] / sum(cm[k]) if sum(cm[k]) else 0.0 for k in range(K)]
    acc = sum(cm[k][k] for k in range(K)) / len(answers)
    present = [k for k in range(K) if sum(cm[k])]
    return sum(recalls[k] for k in present) / len(present), acc, recalls, cm


def macro_f1(cm):
    """7 类各自的 F1（precision 和 recall 的调和平均）再取平均；答案里没有的类不算。"""
    f1s = []
    for k in range(K):
        tp, n_true, n_pred = cm[k][k], sum(cm[k]), sum(cm[j][k] for j in range(K))
        if not n_true:
            continue
        f1s.append(2 * tp / (n_true + n_pred) if n_true + n_pred else 0.0)
    return sum(f1s) / len(f1s)


def main():
    args = sys.argv[1:]
    check_only = "--check" in args
    args = [a for a in args if a != "--check"]
    if not args:
        print(__doc__)
        sys.exit(1)
    here = Path(__file__).resolve().parent
    if len(args) > 1:
        ans_path = Path(args[1])
    elif check_only:   # 查格式不需要答案：按旁边的 sample_submission.csv，没有就按 test/ 里的图片文件名
        ans_path = here / "sample_submission.csv"
        if not ans_path.exists() and (here / "test").is_dir():
            ans_path = here / "test"
    else:
        ans_path = here / "answers.csv"
    if not ans_path.exists():
        print(f"找不到 {ans_path}，请把它的路径写在第二个参数。\n" + __doc__)
        sys.exit(1)
    if ans_path.is_dir():
        answers = {f.stem: 0 for f in sorted(ans_path.glob("*.png"))}
    else:
        answers = read_answers(ans_path)
    notes = []
    try:
        preds = read_submission(args[0], answers.keys(), notes)
    except 格式错误 as e:
        print("格式有问题，读不出来：\n" + str(e))
        sys.exit(2)
    if notes:
        print("能读出来，但有几处和 sample_submission.csv 不一样（已按下面说的自动处理，最好还是改成标准格式）：")
        print("".join(f"  - {s}\n" for s in notes))
    if check_only:
        if not notes:
            print(f"格式没问题：{len(preds)} 行，id 和 {'test/ 里的图片' if ans_path.is_dir() else ans_path.name} 一一对上，label 都是 0–6。")
        return
    bacc, acc, recalls, cm = score(preds, answers)
    print(f"对照 {ans_path.name}（{len(answers)} 张图）算出的几项常用指标：")
    print(f"  accuracy           {acc:.4f}   全部图里猜对的比例")
    print(f"  balanced accuracy  {bacc:.4f}   7 类各自 recall 的平均")
    print(f"  macro-F1           {macro_f1(cm):.4f}   7 类各自 F1 的平均")
    print("\n每类 recall（这一类的图里猜对的比例）：")
    for k in range(K):
        print(f"  {k} {NAMES[k]:<12}{recalls[k]:.3f}  （{cm[k][k]}/{sum(cm[k])}）")
    print("\n混淆矩阵（行 = 真实类别，列 = 预测类别）：")
    print("      " + "".join(f"{j:>6}" for j in range(K)))
    for k in range(K):
        print(f"  {k}   " + "".join(f"{v:>6}" for v in cm[k]))


if __name__ == "__main__":
    main()
