"""
count_useful_tiles.py — Counts tiles with actual content vs empty vegetation
Run: python scripts/count_useful_tiles.py
"""

import cv2
import os
import numpy as np

TILE_DIR = "data/processed/tiles"

total       = 0
useful      = 0
vegetation  = 0
empty       = 0
useful_list = []

for f in sorted(os.listdir(TILE_DIR)):
    if not f.endswith(".png"):
        continue
    total += 1
    img = cv2.imread(os.path.join(TILE_DIR, f))
    if img is None:
        empty += 1
        continue

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    r = img_rgb[:,:,0].astype(float)
    g = img_rgb[:,:,1].astype(float)
    b = img_rgb[:,:,2].astype(float)

    # NDVI-like index: high green relative to red = vegetation
    green_dominance = (g - r) / (g + r + 1e-5)
    veg_ratio = (green_dominance > 0.05).mean()

    # Brightness std — low std = uniform/empty
    brightness = img_rgb.mean(axis=2)
    std = brightness.std()

    # Gray ratio — man-made structures tend to be grey
    gray_mask = (np.abs(r - g) < 20) & (np.abs(g - b) < 20) & (brightness > 80)
    gray_ratio = gray_mask.mean()

    # A tile is "useful" if it has:
    # - Low vegetation dominance OR significant gray (man-made) content
    # - Sufficient texture (std > 15)
    has_manmade = gray_ratio > 0.05
    has_texture = std > 15
    is_mostly_veg = veg_ratio > 0.7

    if has_texture and (has_manmade or not is_mostly_veg):
        useful += 1
        useful_list.append(f)
    elif is_mostly_veg:
        vegetation += 1
    else:
        empty += 1

print(f"Total tiles:       {total}")
print(f"Useful tiles:      {useful}  ← annotate these")
print(f"Vegetation only:   {vegetation}  ← skip these")
print(f"Empty/dark:        {empty}  ← skip these")
print(f"\nUseful tile ratio: {useful/total*100:.1f}%")
print(f"\nFirst 20 useful tiles:")
for t in useful_list[:20]:
    print(f"  {t}")

# Save full useful list
with open("outputs/useful_tiles.txt", "w") as out:
    out.write("\n".join(useful_list))
print(f"\nFull list saved to outputs/useful_tiles.txt")
print(f"You need to annotate {max(0, 800 - useful)} more tiles if useful < 800")