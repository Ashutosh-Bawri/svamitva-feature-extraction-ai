"""
train_final.py — High-accuracy segmentation training for SVAMITVA drone imagery
Place in: svamitva_ai_project/scripts/train_final.py
"""

import os
import torch
import torch.nn as nn
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2

print("Imports loaded. Setting up transforms...")

train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
    A.HueSaturationValue(p=0.3),
    A.GaussianBlur(blur_limit=(3, 5), p=0.2),
    A.GaussNoise(p=0.2),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

val_transform = A.Compose([
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

class DroneDataset(Dataset):
    def __init__(self, image_paths, label_paths, transform=None):
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.transform   = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.image_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        lbl = cv2.imread(self.label_paths[idx], cv2.IMREAD_GRAYSCALE)
        lbl = np.clip(lbl, 0, 3).astype(np.uint8)
        if self.transform:
            aug = self.transform(image=img, mask=lbl)
            img, lbl = aug["image"], aug["mask"]
        else:
            img = torch.tensor(img).permute(2, 0, 1).float() / 255.0
            lbl = torch.tensor(lbl)
        return img, lbl.long()

TILE_DIR        = "data/processed/tiles"
CLEAN_LABEL_DIR = "data/processed/labels_clean"

image_paths, label_paths = [], []
for f in sorted(os.listdir(CLEAN_LABEL_DIR)):
    if not f.endswith("_label.png"):
        continue
    tile_name  = f.replace("_label.png", ".png")
    tile_path  = os.path.join(TILE_DIR, tile_name)
    label_path = os.path.join(CLEAN_LABEL_DIR, f)
    if os.path.exists(tile_path):
        lbl = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        if lbl is not None and lbl.max() > 0:
            image_paths.append(tile_path)
            label_paths.append(label_path)

print(f"Usable clean pairs: {len(image_paths)}")

tr_imgs, va_imgs, tr_lbls, va_lbls = train_test_split(
    image_paths, label_paths, test_size=0.2, random_state=42)

print(f"Train: {len(tr_imgs)} | Val: {len(va_imgs)}")

train_loader = DataLoader(
    DroneDataset(tr_imgs, tr_lbls, train_transform),
    batch_size=4, shuffle=True, num_workers=0, pin_memory=True)

val_loader = DataLoader(
    DroneDataset(va_imgs, va_lbls, val_transform),
    batch_size=4, shuffle=False, num_workers=0, pin_memory=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

print("Loading model weights...")
model = smp.DeepLabV3Plus(
    encoder_name    = "efficientnet-b4",
    encoder_weights = "imagenet",
    in_channels     = 3,
    classes         = 4,
    activation      = None
).to(device)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {total_params / 1e6:.1f}M")
print("Model ready. Starting training...\n")

dice_loss  = smp.losses.DiceLoss(mode="multiclass", smooth=1.0)
focal_loss = smp.losses.FocalLoss(mode="multiclass", gamma=2.0)

def combined_loss(pred, target):
    return dice_loss(pred, target) * 0.6 + focal_loss(pred, target) * 0.4

optimizer = torch.optim.AdamW([
    {"params": model.encoder.parameters(),           "lr": 5e-5},
    {"params": model.decoder.parameters(),           "lr": 2e-4},
    {"params": model.segmentation_head.parameters(), "lr": 2e-4},
], weight_decay=1e-4)

scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=20, T_mult=2, eta_min=1e-6)

def compute_metrics(preds, labels, num_classes=4):
    preds  = preds.cpu().numpy().flatten()
    labels = labels.cpu().numpy().flatten()
    ious, f1s = [], []
    for cls in range(num_classes):
        tp    = ((preds == cls) & (labels == cls)).sum()
        fp    = ((preds == cls) & (labels != cls)).sum()
        fn    = ((preds != cls) & (labels == cls)).sum()
        union = tp + fp + fn
        if union > 0:
            ious.append(tp / union)
        denom = 2 * tp + fp + fn
        if denom > 0:
            f1s.append(2 * tp / denom)
    return (np.mean(ious) if ious else 0.0,
            np.mean(f1s)  if f1s  else 0.0)

EPOCHS   = 80
best_iou = 0.0
os.makedirs("models",  exist_ok=True)
os.makedirs("outputs", exist_ok=True)
log_lines = []

for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss = 0.0
    for imgs, lbls in train_loader:
        imgs, lbls = imgs.to(device), lbls.to(device)
        preds = model(imgs)
        loss  = combined_loss(preds, lbls)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        train_loss += loss.item()

    model.eval()
    val_loss   = 0.0
    all_preds  = []
    all_labels = []
    with torch.no_grad():
        for imgs, lbls in val_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            preds = model(imgs)
            val_loss += combined_loss(preds, lbls).item()
            all_preds.append(preds.argmax(dim=1))
            all_labels.append(lbls)

    all_preds  = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    train_loss /= len(train_loader)
    val_loss   /= len(val_loader)
    val_iou, val_f1 = compute_metrics(all_preds, all_labels)
    scheduler.step()

    line = (f"Epoch {epoch:03d}/{EPOCHS} | "
            f"Train: {train_loss:.4f} | "
            f"Val: {val_loss:.4f} | "
            f"IoU: {val_iou:.4f} | "
            f"F1: {val_f1:.4f}")
    print(line)
    log_lines.append(line)

    if val_iou > best_iou:
        best_iou = val_iou
        torch.save({
            "epoch":       epoch,
            "model_state": model.state_dict(),
            "optimizer":   optimizer.state_dict(),
            "iou":         best_iou,
        }, "models/best_model_final.pth")
        print(f"  ✓ Best model saved (IoU: {best_iou:.4f})")

    if epoch % 20 == 0:
        preds_np  = all_preds.cpu().numpy().flatten()
        labels_np = all_labels.cpu().numpy().flatten()
        names = ["background", "building", "road", "water"]
        parts = []
        for cls in range(4):
            inter = ((preds_np == cls) & (labels_np == cls)).sum()
            union = ((preds_np == cls) | (labels_np == cls)).sum()
            iou   = inter / union if union > 0 else None
            parts.append(f"{names[cls]}: {iou:.3f}" if iou is not None else f"{names[cls]}: N/A")
        print(f"  Per-class: {' | '.join(parts)}")

with open("outputs/training_log.txt", "w") as f:
    f.write("\n".join(log_lines))
    f.write(f"\n\nBest IoU: {best_iou:.4f}")

print(f"\n{'='*60}")
print(f"Training complete. Best IoU: {best_iou:.4f}")
print(f"Model saved: models/best_model_final.pth")
print(f"{'='*60}")