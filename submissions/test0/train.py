#!/usr/bin/env python3
"""快速版：ResNet18 + finetune，2 epoch 跑完"""

import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from pathlib import Path
import sys

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"用 {DEVICE}")

# ============ 数据加载 ============
class DermaDataset(Dataset):
    def __init__(self, folder, labels_csv, transform=None):
        self.folder = Path(folder)
        self.transform = transform
        self.samples = []

        with open(labels_csv, encoding='utf-8-sig') as f:
            for row in csv.DictReader(f):
                img_path = self.folder / row['filename']
                label = int(row['label'])
                self.samples.append((img_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        from PIL import Image
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

# 数据增强 + 标准化
train_transform = transforms.Compose([
    transforms.RandomRotation(20),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

print("加载数据中...")
train_ds = DermaDataset('train', 'train/labels.csv', train_transform)
val_ds = DermaDataset('val', 'val/labels.csv', val_transform)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)
print(f"train: {len(train_ds)}, val: {len(val_ds)}")

# ============ 模型 ============
print("加载 ResNet18...")
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(512, 7)  # 改最后一层
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# ============ 训练 ============
def train_epoch(loader, ep):
    model.train()
    total_loss = 0
    for batch_idx, (x, y) in enumerate(loader):
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if (batch_idx + 1) % 50 == 0:
            print(f"  [{batch_idx+1}/{len(loader)}] loss={loss.item():.4f}")
    return total_loss / len(loader)

def eval_epoch(loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            preds = logits.argmax(1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total

print("\n训练中...")
for ep in range(2):
    train_loss = train_epoch(train_loader, ep)
    val_acc = eval_epoch(val_loader)
    print(f"Epoch {ep}: train_loss={train_loss:.4f}, val_acc={val_acc:.4f}")

print("\n保存模型...")
torch.save(model.state_dict(), 'model.pth')
print("done")
