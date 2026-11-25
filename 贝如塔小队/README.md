# DermaMNIST 皮肤镜病变七分类

模型名：RBF 核支持向量机 + 多尺度颜色与梯度特征 + 类别逆频率平方根加权

使用 MedCraft 赛题数据包。我们对 28×28 的 RGB 图像提取整体颜色直方图，以及不同尺度下的分块颜色统计和梯度方向直方图。特征标准化后，用 RBF 核支持向量机分类。

训练集各类数量差异较大，因此比较三种类别权重。权重等于逆频率权重的 0、0.5 或 1 次幂，以验证集 macro-F1 选出模型，再按官方样例的 ID 顺序生成测试集预测。

## 复现

需要 Python 3.10 或以上，以及 NumPy、Pillow、scikit-learn。在本文件夹运行，将路径替换成解压后的赛题数据目录。

```bash
python train.py /path/to/DermaMNIST --val-output val_predictions.csv
python /path/to/DermaMNIST/score.py --check submission.csv
python /path/to/DermaMNIST/score.py val_predictions.csv /path/to/DermaMNIST/val/labels.csv
```

## 验证结果

选中的模型使用逆频率权重的平方根。官方脚本在 1003 张验证图上给出准确率 74.98%、平衡准确率 43.33%、macro-F1 45.94%。黑素细胞痣识别了 605/671 张，皮肤纤维瘤识别了 1/12 张。测试集预测包含 2005 行，已通过官方格式检查。
