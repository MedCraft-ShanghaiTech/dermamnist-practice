# test0 - ResNet18 基础版

## 方法
- 预训练 ResNet18（ImageNet 权重）
- 最后一层改成 7 分类
- 2 epoch 微调（learning rate 1e-4）
- 数据增强：旋转、翻转、色彩抖动

## 结果
- Validation accuracy: 77.37%
- Balanced accuracy: 41.0%
- Macro-F1: 45.1%

## 注意
这是快速版本，用来测试流程。后续会优化类别不平衡问题。
