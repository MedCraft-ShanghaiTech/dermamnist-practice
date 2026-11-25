#!/usr/bin/env python3
"""推理 test/，输出 submission.csv"""

import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from pathlib import Path
from PIL import Image
import csv

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"用 {DEVICE}")

# 加载模型
print("加载模型...")
model = models.resnet18(weights=None)
model.fc = nn.Linear(512, 7)
model.load_state_dict(torch.load('model.pth', map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

# 推理变换
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

# 推理
print("推理 test/...")
test_dir = Path('test')
test_files = sorted([f for f in os.listdir(test_dir) if f.endswith('.png')])
print(f"找到 {len(test_files)} 张图")

predictions = []
with torch.no_grad():
    for i, fname in enumerate(test_files):
        if (i + 1) % 500 == 0:
            print(f"  [{i+1}/{len(test_files)}]")

        img_path = test_dir / fname
        img = Image.open(img_path).convert('RGB')
        img_t = transform(img).unsqueeze(0).to(DEVICE)

        logits = model(img_t)
        pred = logits.argmax(1).item()

        # id 是去掉 .png 的文件名
        img_id = fname[:-4]
        predictions.append((img_id, pred))

# 保存
print("保存 submission.csv...")
with open('submission.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['id', 'label'])
    for img_id, pred in predictions:
        writer.writerow([img_id, pred])

print(f"完成！{len(predictions)} 行写入 submission.csv")
