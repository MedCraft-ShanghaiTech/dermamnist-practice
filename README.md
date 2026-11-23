# 实践题：皮肤镜图片 7 分类（DermaMNIST）

**任务**：用 `train/` 里带答案的皮肤镜小图训练一个模型，预测 `test/` 里每张图属于 7 类皮肤病变中的哪一类，存成 `submission.csv` 交上来。数据长什么样、有哪些类别、提交格式是什么，看 `sample_submission.csv` 和 `train/`、`val/` 里的 `labels.csv`，让 AI 去读。

**截止时间**：【现场公布】

## 怎么交

1. fork 组织者的 GitHub 仓库：https://github.com/MedCraft-ShanghaiTech/dermamnist-practice
2. 在 fork 出来的仓库里建文件夹 `submissions/队名/`，文件夹名就是队名（榜上只显示它，建议用英文或拼音、不带空格），里面放：`submission.csv`、你们的代码、一个 `README.md` 写几句"我们怎么做的"。数据和模型权重不要放。
3. 推上去，再向组织者仓库提一个 Pull Request（PR），PR 就是提交。推送和提 PR 都可以让 AI 帮你做（比如用 `gh` 命令）。截止前可以接着改、接着推，以最新一次为准。
4. 实在提不了 PR：把 `submission.csv` 改名成"队名.csv"，微信发给工作组。

## 怎么打分

截止后统一打分，投屏一张榜，只显示队名，不评奖。指标是**平衡准确率**（balanced accuracy）：每一类各自猜对的比例，再把 7 类直接平均。不用普通准确率，是因为约 67% 的图是"黑素细胞痣"，什么都不学、全猜痣，普通准确率就有约 67%，平衡准确率却只有 1/7 ≈ 0.14。数据包里的 `score.py` 和组织者打分用的是同一份，可以用它自查。

## 小贴士

1. **只用发给你的文件**，别让 AI 去下载 MedMNIST 原版：原版带测试集答案，用了就白练了。
2. 先交一版几分钟能跑完的，再慢慢提高；训练跑太久就停掉，换小一点的。
3. 跟 AI 说"代码和结果都存在当前文件夹"，不然它可能存到你找不到的地方。
4. 文件夹和文件名都用英文，别带中文和空格。
5. 别用 Excel 打开 CSV 再保存，格式会坏；想看就用 Trae 或记事本打开。
6. 图里中文变成方块，就让 AI 换个能显示中文的字体，或者图里写英文。
7. 验证集分数不等于测试集分数：小类图很少，差 0.03 很正常。

## 出处与许可

数据来自 MedMNIST v2 的 DermaMNIST 子集（原图来自 HAM10000），这里转成了 PNG + CSV，测试集已打乱、重新编号。许可 CC BY-NC 4.0，仅限非商业用途，使用请引用：Yang J, et al. MedMNIST v2. *Scientific Data*, 2023；Tschandl P, Rosendahl C, Kittler H. The HAM10000 dataset. *Scientific Data* 5, 180161 (2018)。
