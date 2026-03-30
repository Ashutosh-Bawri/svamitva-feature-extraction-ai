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

# ── Augmentation ──────────────────────────────────────────────────────────────
train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.GaussianBlur(p=0.2),
    A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
    ToTensorV2()
])

val_transform = A.Compose([
    A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
    ToTensorV2()
])

# ── Dataset ───────────────────────────────────────────────────────────────────
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
        lbl = np.clip(lbl, 0, 3)

        if self.transform:
            aug = self.transform(image=img, mask=lbl)
            img, lbl = aug["image"], aug["mask"]
        else:
            img = torch.tensor(img).permute(2,0,1).float() / 255.0
            lbl = torch.tensor(lbl)

        return img, lbl.long()

# ── Data ──────────────────────────────────────────────────────────────────────
TILE_DIR  = "data/processed/tiles"
LABEL_DIR = "data/processed/labels"

image_paths, label_paths = [], []
for f in sorted(os.listdir(LABEL_DIR)):
    if not f.endswith("_label.png"):
        continue
    tile_name  = f.replace("_label.png", ".png")
    tile_path  = os.path.join(TILE_DIR,  tile_name)
    label_path = os.path.join(LABEL_DIR, f)
    if os.path.exists(tile_path):
        image_paths.append(tile_path)
        label_paths.append(label_path)

print(f"Total pairs: {len(image_paths)}")

tr_imgs, va_imgs, tr_lbls, va_lbls = train_test_split(
    image_paths, label_paths, test_size=0.2, random_state=42)

train_loader = DataLoader(
    DroneDataset(tr_imgs, tr_lbls, train_transform),
    batch_size=8, shuffle=True, num_workers=0, pin_memory=True)

val_loader = DataLoader(
    DroneDataset(va_imgs, va_lbls, val_transform),
    batch_size=8, shuffle=False, num_workers=0, pin_memory=True)

print(f"Train: {len(tr_imgs)} | Val: {len(va_imgs)}")

# ── Model ─────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

model = smp.DeepLabV3Plus(
    encoder_name="resnet50",
    encoder_weights="imagenet",
    in_channels=3,
    classes=4
).to(device)

# ── Loss ──────────────────────────────────────────────────────────────────────
dice_loss = smp.losses.DiceLoss(mode="multiclass")
ce_loss   = nn.CrossEntropyLoss(
    weight=torch.tensor([0.5, 2.0, 2.0, 3.0]).to(device)
)

def combined_loss(pred, target):
    return dice_loss(pred, target) + ce_loss(pred, target)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=50, eta_min=1e-6)

# ── IoU ───────────────────────────────────────────────────────────────────────
def compute_iou(preds, labels, num_classes=4):
    ious = []
    preds  = preds.cpu().numpy().flatten()
    labels = labels.cpu().numpy().flatten()
    for cls in range(num_classes):
        inter = ((preds==cls) & (labels==cls)).sum()
        union = ((preds==cls) | (labels==cls)).sum()
        if union > 0:
            ious.append(inter / union)
    return np.mean(ious) if ious else 0.0

# ── Training ──────────────────────────────────────────────────────────────────
EPOCHS   = 50
best_iou = 0.0
os.makedirs("models", exist_ok=True)

for epoch in range(1, EPOCHS+1):
    model.train()
    train_loss = 0.0
    for imgs, lbls in train_loader:
        imgs, lbls = imgs.to(device), lbls.to(device)
        preds = model(imgs)
        loss  = combined_loss(preds, lbls)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    model.eval()
    val_loss, val_iou = 0.0, 0.0
    with torch.no_grad():
        for imgs, lbls in val_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            preds     = model(imgs)
            val_loss += combined_loss(preds, lbls).item()
            val_iou  += compute_iou(preds.argmax(dim=1), lbls)

    train_loss /= len(train_loader)
    val_loss   /= len(val_loader)
    val_iou    /= len(val_loader)
    scheduler.step()

    print(f"Epoch {epoch:02d}/{EPOCHS} | "
          f"Train: {train_loss:.4f} | "
          f"Val: {val_loss:.4f} | "
          f"IoU: {val_iou:.4f}")

    if val_iou > best_iou:
        best_iou = val_iou
        torch.save(model.state_dict(), "models/best_model.pth")
        print(f"  ✓ Best model saved (IoU: {best_iou:.4f})")

print(f"\nDone. Best IoU: {best_iou:.4f}")