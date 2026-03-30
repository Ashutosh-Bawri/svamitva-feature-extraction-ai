import cv2
import os
import numpy as np
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

MODEL_TYPE = "vit_h"
CHECKPOINT = "sam_vit_h_4b8939.pth"

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT)
sam.to(device)

mask_generator = SamAutomaticMaskGenerator(
    sam,
    points_per_side=32,
    pred_iou_thresh=0.88,
    stability_score_thresh=0.92,
    min_mask_region_area=100
)

INPUT_DIR = "data/processed/tiles"
MASK_DIR = "data/processed/masks"
LABEL_DIR = "data/processed/labels"

os.makedirs(MASK_DIR, exist_ok=True)
os.makedirs(LABEL_DIR, exist_ok=True)

files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".png")]
print(f"Processing {len(files)} tiles...")

for i, file in enumerate(files):
    path = os.path.join(INPUT_DIR, file)
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if image is None or image.mean() < 5:
        continue

    masks = mask_generator.generate(image)

    if len(masks) == 0:
        continue

    label_mask = np.zeros((512, 512), dtype=np.uint8)

    for mask in masks:
        seg = mask["segmentation"]
        area = mask["area"]
        bbox = mask["bbox"]
        w = bbox[2]
        h = bbox[3]
        aspect = w / (h + 1e-5)

        # Get region color stats
        region_pixels = image[seg]
        mean_color = region_pixels.mean(axis=0)
        r, g, b = mean_color
        std_color = region_pixels.std()

        # Skip very large masks (likely entire background)
        if area > 80000:
            continue

        # Skip very small noise masks
        if area < 150:
            continue

        # BUILDING detection
        # Buildings: medium-large area, rectangular, man-made colors
        # Tin roofs = silvery/grey, RCC = grey/white, tiled = red-brown
        is_rectangular = 0.2 < aspect < 5.0
        is_medium = 500 < area < 35000
        is_manmade_color = (
            (r > 100 and g > 100 and b > 100 and std_color < 40) or  # grey/silver tin roof
            (r > g + 15 and r > b + 15 and area < 20000) or           # reddish tiled roof
            (r > 150 and g > 150 and b > 150 and std_color < 30)      # white/bright RCC
        )

        if is_medium and is_rectangular and is_manmade_color:
            label_mask[seg] = 1
            continue

        # WATER detection
        # Water: blue dominant, smooth texture
        is_blue = b > r + 10 and b > g + 5
        is_smooth = std_color < 25
        if is_blue and is_smooth and area > 1000:
            label_mask[seg] = 3
            continue

        # ROAD detection
        # Roads: elongated shape, earthy/grey color, medium-large area
        is_elongated = aspect > 3.0 or aspect < 0.33
        is_earthy = (
            (r > g and r > b and 80 < r < 200) or   # brown/red dirt road
            (abs(r-g) < 20 and abs(g-b) < 20 and r < 150)  # grey road
        )
        is_road_size = 2000 < area < 60000

        if is_elongated and is_earthy and is_road_size:
            label_mask[seg] = 2

    # Save label
    label_path = os.path.join(LABEL_DIR, file.replace(".png", "_label.png"))
    cv2.imwrite(label_path, label_mask)

    # Save overlay
    overlay = image.copy()
    colors = {1: (255, 0, 0), 2: (0, 255, 0), 3: (0, 0, 255)}
    for class_id, color in colors.items():
        overlay[label_mask == class_id] = (
            overlay[label_mask == class_id] * 0.4 +
            np.array(color) * 0.6
        ).astype(np.uint8)

    overlay_path = os.path.join(MASK_DIR, file.replace(".png", "_overlay.png"))
    cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    b_count = (label_mask == 1).sum()
    r_count = (label_mask == 2).sum()
    w_count = (label_mask == 3).sum()
    print(f"[{i+1}/{len(files)}] {file} — "
          f"buildings: {b_count}, roads: {r_count}, water: {w_count}")

print("\nDone!")