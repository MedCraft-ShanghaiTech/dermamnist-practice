# TeamVitality · DermaMNIST 7 分类

## 方案

轻量基线，不依赖 GPU，单个 CPU 上约 3 分钟跑完：

1. **特征**：每张 28×28 彩色图提取 195 维特征 —— 三个颜色通道各 16 bin 的归一化直方图（48 维）+ 7×7 平均池化缩略图（147 维）。28×28 原图信息量有限，直接把 2352 维像素丢给模型容易过拟合，先用低维的颜色+粗略空间特征。
2. **类别不平衡**：训练集里黑素细胞痣占比 67%（4693/7007），最少的一类只有 80 张。对少数类做随机过采样，把 7 类补齐到同一数量再训练，避免模型退化成"全猜痣"。
3. **模型**：标准化 + 两层 MLP（512-256），早停。

## 自查结果（val，1003 张）

| 指标 | 数值 |
|---|---|
| accuracy | 0.7198 |
| balanced accuracy | 0.4957 |
| macro-F1 | 0.5094 |

对照：不做类别均衡时 accuracy 0.7537 但 macro-F1 只有 0.4305，最终选择了指标更均衡的版本。

## 文件

- `submission.csv`：最终提交（2005 行，已用 `score.py --check` 校验）
- `train_baseline.py`：训练 + 预测脚本，`python train_baseline.py` 即可复现

## 复现

```bash
pip install numpy scikit-learn pillow
python train_baseline.py   # 读取同级 DermaMNIST/ 目录，输出 submission.csv
```
