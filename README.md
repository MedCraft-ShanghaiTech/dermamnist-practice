# 实践题：皮肤镜图片 7 分类（DermaMNIST）

**任务**：用 `train/` 训练模型，预测 `test/` 里每张图的类别，存成 `submission.csv`（格式同 `sample_submission.csv`）。只用这里给的文件，不要下载 MedMNIST 原版（原版带测试集答案）。

**截止**：【现场公布】

**怎么交**：fork https://github.com/MedCraft-ShanghaiTech/dermamnist-practice ，把 `submission.csv`、代码和几句"我们怎么做的"放进 `submissions/队名/`，再提一个 Pull Request。提不了的话，把 `submission.csv` 改名成"队名.csv"微信发给工作组。

**怎么打分**：平衡准确率（balanced accuracy，每一类各自猜对的比例再取平均）。截止后统一打分投屏，只显示队名，不评奖。`score.py` 可以自查。

**数据许可**：MedMNIST v2 的 DermaMNIST 子集（原图来自 HAM10000），CC BY-NC 4.0，仅限非商业用途。引用：Yang J, et al. MedMNIST v2. *Scientific Data*, 2023；Tschandl P, et al. The HAM10000 dataset. *Scientific Data* 5, 180161 (2018)。
