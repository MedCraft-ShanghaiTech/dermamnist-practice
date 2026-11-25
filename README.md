# 实践题：皮肤镜图片 7 分类（DermaMNIST）

**任务**：用 `train/` 训练模型，预测 `test/` 里每张图的类别，存成 `submission.csv`。只用这里给的文件，不要下载 MedMNIST 原版（原版带测试集答案）。

**提交格式**：`submission.csv` 和 `sample_submission.csv` 一样，两列 `id,label`；`test/` 里每张图一行，共 2005 行；`id` 是图片文件名去掉 `.png`（如 `test_0000`），`label` 填 0–6 的类别编号。交之前在解压出的 `DermaMNIST/` 文件夹里运行 `python score.py --check submission.csv` 检查格式。

**截止**：【现场公布】

**怎么交**：fork https://github.com/MedCraft-ShanghaiTech/dermamnist-practice ，在 `submissions/队名/` 里放 `submission.csv`、代码和一个 `README.md`（几句"我们怎么做的"），不放模型权重和数据，再提一个 Pull Request。提不了的话，把 `submission.csv` 改名成"队名.csv"微信发给工作组。

**结果**：截止后统一打分投屏，只显示队名，不评奖。`score.py` 也能在 `val/` 上算几项常用指标，用法见文件开头。

**建议**：先搞清任务，跑通再改进；省着用积分。

**数据许可**：MedMNIST v2 的 DermaMNIST 子集（原图来自 HAM10000），CC BY-NC 4.0，仅限非商业用途。引用：Yang J, et al. MedMNIST v2. *Scientific Data*, 2023；Tschandl P, et al. The HAM10000 dataset. *Scientific Data* 5, 180161 (2018)。
