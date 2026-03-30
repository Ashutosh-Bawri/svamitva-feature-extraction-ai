"""
audit_tiles.py — Visual audit of tile quality per village
Run: python scripts/audit_tiles.py
"""

import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

TILE_DIR = "data/processed/tiles"

# Analyze each tile
stats = {}
for f in sorted(os.listdir(TILE_DIR)):
    if not f.endswith(".png"):
        continue

    img = cv2.imread(os.path.join(TILE_DIR, f))
    if img is None:
        continue

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    r = img_rgb[:,:,0].astype(float)
    g = img_rgb[:,:,1].astype(float)
    b = img_rgb[:,:,2].astype(float)

    brightness  = img_rgb.mean(axis=2)
    std         = brightness.std()
    mean_bright = brightness.mean()

    # Gray (manmade) ratio
    gray_mask  = (np.abs(r-g) < 25) & (np.abs(g-b) < 25) & (brightness > 60)
    gray_ratio = gray_mask.mean()

    # Green (vegetation) ratio
    green_dom  = (g > r + 10) & (g > b + 10)
    green_ratio= green_dom.mean()

    # Distorted = very dark or very uniform
    is_dark      = mean_bright < 30
    is_uniform   = std < 12
    is_distorted = is_dark or is_uniform

    # Classification
    if is_distorted:
        quality = "distorted"
    elif gray_ratio > 0.08:
        quality = "useful"      # has manmade structures
    elif green_ratio > 0.6:
        quality = "vegetation"
    else:
        quality = "useful"      # mixed, still annotatable

    # Group by village
    village = f.split("_hd.tif")[0] if "_hd" in f else f.split(".tif")[0]
    if village not in stats:
        stats[village] = {"useful":0, "vegetation":0, "distorted":0, "total":0}
    stats[village][quality] += 1
    stats[village]["total"]  += 1

# Print summary
print(f"\n{'Village':<20} {'Total':>6} {'Useful':>7} {'Veg':>6} {'Distorted':>10} {'Useful%':>8}")
print("-" * 65)

grand_useful = grand_veg = grand_dist = grand_total = 0
for v in sorted(stats.keys()):
    s = stats[v]
    pct = s["useful"] / s["total"] * 100 if s["total"] > 0 else 0
    print(f"{v:<20} {s['total']:>6} {s['useful']:>7} {s['vegetation']:>6} {s['distorted']:>10} {pct:>7.0f}%")
    grand_useful += s["useful"]
    grand_veg    += s["vegetation"]
    grand_dist   += s["distorted"]
    grand_total  += s["total"]

print("-" * 65)
pct = grand_useful / grand_total * 100 if grand_total > 0 else 0
print(f"{'TOTAL':<20} {grand_total:>6} {grand_useful:>7} {grand_veg:>6} {grand_dist:>10} {pct:>7.0f}%")

print(f"\nConclusion:")
if grand_useful >= 600:
    print(f"  You have {grand_useful} useful tiles — enough for 95% IoU.")
    print(f"  Focus annotation on useful tiles only.")
elif grand_useful >= 400:
    print(f"  You have {grand_useful} useful tiles — enough for ~85% IoU.")
    print(f"  Export 5 more village areas from QGIS at 600 DPI to get to 600+.")
else:
    print(f"  Only {grand_useful} useful tiles — need more QGIS exports.")
    print(f"  Export 10 more village areas from QGIS at 600 DPI.")

# Show sample tiles per village for visual check
print(f"\nGenerating visual sample grid...")
os.makedirs("outputs", exist_ok=True)

villages = sorted(stats.keys())
fig, axes = plt.subplots(len(villages), 5, figsize=(20, 4 * len(villages)))
if len(villages) == 1:
    axes = axes.reshape(1, -1)

for vi, village in enumerate(villages):
    # Get 5 sample tiles from this village
    village_tiles = [f for f in os.listdir(TILE_DIR)
                     if f.endswith(".png") and
                     (f.split("_hd.tif")[0] if "_hd" in f else f.split(".tif")[0]) == village]
    samples = village_tiles[::max(1, len(village_tiles)//5)][:5]

    for ti in range(5):
        ax = axes[vi][ti]
        if ti < len(samples):
            img = cv2.imread(os.path.join(TILE_DIR, samples[ti]))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax.imshow(img)
            ax.set_title(samples[ti][-20:], fontsize=6)
        ax.axis("off")

    axes[vi][0].set_ylabel(village, fontsize=9, rotation=0,
                            labelpad=80, va="center")

plt.suptitle("Sample Tiles Per Village — Check Visual Quality", fontsize=13)
plt.tight_layout()
plt.savefig("outputs/tile_audit.png", dpi=100, bbox_inches="tight")
plt.show()
print("Saved: outputs/tile_audit.png")