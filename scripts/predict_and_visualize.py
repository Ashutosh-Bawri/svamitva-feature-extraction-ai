import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import segmentation_models_pytorch as smp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ================= CONFIG =================
BASE = r"C:\Users\ashut\OneDrive\Desktop\svamitva_ai_project"
TILES_DIR = os.path.join(BASE, "data", "processed", "tiles")
LABELS_DIR = os.path.join(BASE, "data", "processed", "labels_clean")
MODEL_PATH = os.path.join(BASE, "models", "best_model_final.pth")
OUTPUT_DIR = os.path.join(BASE, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

IMG_SIZE = 512
NUM_CLASSES = 4
CLASS_NAMES = ["background", "building", "road", "water"]
CLASS_COLORS = [(50,50,50),(255,100,50),(255,220,50),(50,150,255)]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= MODEL =================
model = smp.DeepLabV3Plus(
    encoder_name="efficientnet-b4",
    encoder_weights=None,
    in_channels=3,
    classes=4
)

# ================= DATASET =================
class TileDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs
        self.img_tf = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])

    def __len__(self): return len(self.pairs)

    def __getitem__(self, idx):
        img_path, lbl_path = self.pairs[idx]

        img = Image.open(img_path).convert("RGB")
        lbl = np.array(Image.open(lbl_path))

        lbl = torch.from_numpy(lbl).long()
        lbl = F.interpolate(lbl.unsqueeze(0).unsqueeze(0).float(),
                            size=(IMG_SIZE, IMG_SIZE),
                            mode='nearest').squeeze().long()

        return self.img_tf(img), lbl, os.path.basename(img_path)

def load_pairs():
    pairs = []

    for f in os.listdir(LABELS_DIR):
        if not f.endswith(".png"):
            continue

        # remove "_label"
        base_name = f.replace("_label", "")

        img_path = os.path.join(TILES_DIR, base_name)
        lbl_path = os.path.join(LABELS_DIR, f)

        if os.path.exists(img_path):
            pairs.append((img_path, lbl_path))

    print(f"Loaded {len(pairs)} matched pairs")
    return pairs

# ================= METRICS =================
def compute_iou(pred, target):
    ious = []
    p = pred.cpu().numpy().flatten()
    t = target.cpu().numpy().flatten()

    for c in range(NUM_CLASSES):
        inter = np.sum((p==c)&(t==c))
        union = np.sum((p==c)|(t==c))
        ious.append(inter/union if union>0 else np.nan)

    return ious

def colorize(mask):
    rgb = np.zeros((*mask.shape,3), dtype=np.uint8)
    for c, col in enumerate(CLASS_COLORS):
        rgb[mask==c] = col
    return rgb

# ================= MAIN =================
def main():
    print("Loading model...")

    model.to(DEVICE)

    # 🔥 FIXED CHECKPOINT LOADING
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)

    if isinstance(ckpt, dict):
        if 'model_state' in ckpt:
            model.load_state_dict(ckpt['model_state'])
        elif 'state_dict' in ckpt:
            model.load_state_dict(ckpt['state_dict'])
        elif 'model' in ckpt:
            model.load_state_dict(ckpt['model'])
        else:
            model.load_state_dict(ckpt)
    else:
        model.load_state_dict(ckpt)

    model.eval()
    print("Model loaded successfully.")

    pairs = load_pairs()
    print(f"Evaluating on {len(pairs)} tiles...")

    dataset = TileDataset(pairs[:200])
    loader = DataLoader(dataset, batch_size=4, shuffle=False)

    all_ious = [[] for _ in range(NUM_CLASSES)]
    vis_items = []

    with torch.no_grad():
        for imgs, lbls, names in loader:
            imgs = imgs.to(DEVICE)
            preds = model(imgs).argmax(dim=1)

            for i in range(len(imgs)):
                ious = compute_iou(preds[i], lbls[i])
                for c, iou in enumerate(ious):
                    if not np.isnan(iou):
                        all_ious[c].append(iou)

                if len(vis_items) < 8:
                    vis_items.append((imgs[i].cpu(), lbls[i].cpu().numpy(),
                                      preds[i].cpu().numpy(), names[i]))

    print("\n===== RESULTS =====")
    mean_ious = []

    for c in range(NUM_CLASSES):
        m = np.nanmean(all_ious[c]) if all_ious[c] else 0
        mean_ious.append(m)
        print(f"{CLASS_NAMES[c]}: {m*100:.2f}%")

    miou = np.nanmean(mean_ious)
    print(f"Mean IoU: {miou*100:.2f}%")

    # ================= VISUALIZATION =================
    n = min(8, len(vis_items))
    fig, axes = plt.subplots(n, 3, figsize=(15, n*4))

    if n == 1: axes = [axes]

    mean = np.array([0.485,0.456,0.406])
    std = np.array([0.229,0.224,0.225])

    for row, (img, gt, pred, name) in enumerate(vis_items):
        img_np = img.permute(1,2,0).numpy()
        img_np = np.clip(img_np*std + mean, 0, 1)

        axes[row][0].imshow(img_np)
        axes[row][1].imshow(colorize(gt))
        axes[row][2].imshow(colorize(pred))

        axes[row][0].set_title("Input")
        axes[row][1].set_title("Ground Truth")
        axes[row][2].set_title("Prediction")

        for j in range(3):
            axes[row][j].axis("off")

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "eval_results.png")
    plt.savefig(out_path)
    plt.close()

    print(f"\nSaved → {out_path}")

if __name__ == "__main__":
    main()