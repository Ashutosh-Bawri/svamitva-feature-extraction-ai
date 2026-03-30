import json
import numpy as np
import cv2
import os
import urllib.parse

ANNOTATIONS_FILE = "data/annotations/annotations.json"
TILE_DIR         = "data/processed/tiles"
OUTPUT_DIR       = "data/processed/labels_clean"

os.makedirs(OUTPUT_DIR, exist_ok=True)

CLASS_MAP = {
    "building":   1,
    "road":       2,
    "water":      3,
    "background": 0
}

with open(ANNOTATIONS_FILE, "r") as f:
    data = json.load(f)

print(f"Total annotations: {len(data)}")
converted = 0

for item in data:
    image_url = item.get("data", {}).get("image", "")
    filename  = os.path.basename(urllib.parse.unquote(image_url))

    # Strip Label Studio UUID prefix e.g. "13ea829b-village1_hd.tif_0.png"
    if "-" in filename:
        parts = filename.split("-", 1)
        if len(parts[0]) == 8 and all(c in "0123456789abcdef" for c in parts[0]):
            filename = parts[1]

    if not filename.endswith(".png"):
        continue

    tile_path = os.path.join(TILE_DIR, filename)
    if not os.path.exists(tile_path):
        print(f"Tile not found: {filename}")
        continue

    img      = cv2.imread(tile_path)
    H, W     = img.shape[:2]
    mask     = np.zeros((H, W), dtype=np.uint8)

    annotations = item.get("annotations", [])
    if not annotations:
        continue

    results = annotations[0].get("result", [])

    for result in results:
        if result.get("type") != "polygonlabels":
            continue

        label_names = result["value"].get("polygonlabels", [])
        if not label_names:
            continue

        label_name = label_names[0].lower()
        class_id   = CLASS_MAP.get(label_name, 0)

        points = result["value"].get("points", [])
        if not points:
            continue

        px   = [(int(p[0] / 100 * W), int(p[1] / 100 * H)) for p in points]
        poly = np.array(px, dtype=np.int32)
        cv2.fillPoly(mask, [poly], class_id)

    out_path = os.path.join(OUTPUT_DIR, filename.replace(".png", "_label.png"))
    cv2.imwrite(out_path, mask)
    converted += 1
    print(f"Converted: {filename} — classes: {np.unique(mask)}")

print(f"\nDone! Converted {converted} annotations")