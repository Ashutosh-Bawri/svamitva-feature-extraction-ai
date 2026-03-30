"""
predict_geojson.py
Runs inference on ALL tiles and exports detected features
(buildings, roads, water) as a GeoJSON FeatureCollection.
This is the "Feature-extracted datasets" deliverable for the hackathon.
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import segmentation_models_pytorch as smp
import cv2

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE        = r"C:\Users\ashut\OneDrive\Desktop\svamitva_ai_project"
TILES_DIR   = os.path.join(BASE, "data", "processed", "tiles")
MODEL_PATH  = os.path.join(BASE, "models", "best_model_final.pth")
OUTPUT_DIR  = os.path.join(BASE, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Config ─────────────────────────────────────────────────────────────────────
IMG_SIZE    = 512
NUM_CLASSES = 4
CLASS_NAMES = ["background", "building", "road", "water"]
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMG_TF = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

# ── Model ──────────────────────────────────────────────────────────────────────
def load_model():
    import segmentation_models_pytorch as smp

    model = smp.DeepLabV3Plus(
        encoder_name="efficientnet-b4",
        encoder_weights=None,
        in_channels=3,
        classes=4
    )

    model.to(DEVICE)

    # 🔥 Load checkpoint correctly
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
    return model
# ── Contour → polygon coords ───────────────────────────────────────────────────
def mask_to_polygons(binary_mask, class_name, tile_name, tile_idx, img_w, img_h):
    """Extract contours from a binary mask and return GeoJSON features."""
    features = []
    # Morphological cleanup
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(binary_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt_idx, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        # Skip tiny noise
        min_area = 100 if class_name == "building" else 50
        if area < min_area:
            continue

        # Simplify contour
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx  = cv2.approxPolyDP(cnt, epsilon, True)
        if len(approx) < 3:
            continue

        # Normalise coords to 0-1 range (relative to tile)
        coords = []
        for pt in approx:
            x = float(pt[0][0]) / img_w
            y = float(pt[0][1]) / img_h
            coords.append([x, y])
        coords.append(coords[0])  # close ring

        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [coords]
            },
            "properties": {
                "class":      class_name,
                "tile":       tile_name,
                "tile_index": tile_idx,
                "area_px":    float(area),
                "source":     "svamitva_ai_model_v1"
            }
        }
        features.append(feature)
    return features

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("Loading model...")
    model = load_model()
    print(f"Model loaded. Running on: {DEVICE}")

    tiles = sorted([f for f in os.listdir(TILES_DIR) if f.endswith(".png")])
    print(f"Processing {len(tiles)} tiles...")

    all_features = []
    class_counts = {n: 0 for n in CLASS_NAMES[1:]}  # skip background

    for idx, tile_name in enumerate(tiles):
        if idx % 100 == 0:
            print(f"  [{idx}/{len(tiles)}] {tile_name}")

        img_path = os.path.join(TILES_DIR, tile_name)
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            continue

        img_w, img_h = img.size
        tensor = IMG_TF(img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = model(tensor)
            pred   = logits.argmax(dim=1).squeeze().cpu().numpy()  # (512,512)

        # Resize pred back to original tile size for accurate polygon coords
        pred_resized = cv2.resize(pred.astype(np.uint8), (img_w, img_h),
                                  interpolation=cv2.INTER_NEAREST)

        # Extract features for each non-background class
        for class_idx, class_name in enumerate(CLASS_NAMES):
            if class_idx == 0:
                continue  # skip background
            binary = (pred_resized == class_idx).astype(np.uint8)
            if binary.sum() == 0:
                continue
            feats = mask_to_polygons(binary, class_name, tile_name, idx, img_w, img_h)
            all_features.extend(feats)
            class_counts[class_name] += len(feats)

    # ── Write GeoJSON ──────────────────────────────────────────────────────────
    geojson = {
        "type": "FeatureCollection",
        "name": "svamitva_extracted_features",
        "crs": {
            "type": "name",
            "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}
        },
        "features": all_features
    }

    out_path = os.path.join(OUTPUT_DIR, "extracted_features.geojson")
    with open(out_path, "w") as f:
        json.dump(geojson, f, indent=2)

    print("\n" + "="*50)
    print("  GEOJSON EXPORT COMPLETE")
    print("="*50)
    print(f"  Total features extracted: {len(all_features)}")
    for cls, cnt in class_counts.items():
        print(f"    {cls:12s}: {cnt} polygons")
    print(f"\n  Output → {out_path}")
    print("  Open this file in QGIS to view extracted features.")
    print("="*50)

if __name__ == "__main__":
    main()