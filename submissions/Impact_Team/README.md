# Impact_Team — DermaMNIST 7 分类

**我们怎么做的：**

- 模型：`torchvision` 的 ResNet18（ImageNet 预训练），把最后一层全连接换成 7 类输出，在 28×28 的皮肤镜图上微调（放大到 96×96 输入）。
- 类别极不均衡（黑素细胞痣 nv 占约 2/3，皮肤纤维瘤 df / 血管病变 vasc 各只有约 100 张），用**类别加权交叉熵**缓解，让小类也能被学到。
- 数据增强：随机裁剪+翻转、颜色抖动；预测时用 **TTA**（中心裁剪 / 水平翻转 / 两种缩放共 4 个视图取平均）。
- 训练分两步：① 在 train 上训练 15 轮，按 val 的 balanced accuracy 选最优模型；② 用该模型热启动，把 val 也并入训练集（全部 8010 张标注图）低学习率再微调 8 轮，作为最终提交模型。

**验证集（val，1003 张）自查指标**（第①步选出最优模型时测得，第②步把 val 并入训练后无独立验证集）：

| 指标 | 值 |
|---|---|
| accuracy | 0.6181 |
| balanced accuracy | 0.6065 |
| macro-F1 | 0.4718 |

**提交文件：** `submission.csv`（两列 `id,label`，2005 行，label 为 0–6）。已经过 `python score.py --check submission.csv` 校验，格式无误。

**复现：** 数据解压到 `DermaMNIST/` 后，在本目录运行：
```bash
pip install torch torchvision numpy pillow
python train.py --data /path/to/DermaMNIST --epochs 15
python train.py --data /path/to/DermaMNIST --resume best_model.pt --combine_val --epochs 8 --lr 3e-4
```
（环境为 macOS + Apple Silicon MPS；GPU/CPU 同样可跑，把 `--data` 改成你的数据路径即可。）

> 按要求，本文件夹只含 `submission.csv`、代码 `train.py` 和本说明，不含模型权重和数据集。
