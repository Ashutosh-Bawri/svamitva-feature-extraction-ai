"""
find_unannotated.py — Shows exactly which tiles still need annotation
Run: python scripts/find_unannotated.py
"""

import os
import cv2
import numpy as np

TILE_DIR        = "data/processed/tiles"
CLEAN_LABEL_DIR = "data/processed/labels_clean"

# Get already annotated tiles
annotated = set()
for f in os.listdir(CLEAN_LABEL_DIR):
    if f.endswith("_label.png"):
        annotated.add(f.replace("_label.png", ".png"))

print(f"Already annotated: {len(annotated)} tiles")

# Find unannotated useful tiles grouped by village
unannotated = {}
for f in sorted(os.listdir(TILE_DIR)):
    if not f.endswith(".png"):
        continue
    if f in annotated:
        continue

    # Check if tile has useful content
    img = cv2.imread(os.path.join(TILE_DIR, f))
    if img is None:
        continue
    if img.std() < 15:
        continue  # skip empty/dark tiles

    # Group by village
    village = f.split("_hd")[0] if "_hd" in f else f.split(".tif")[0]
    if village not in unannotated:
        unannotated[village] = []
    unannotated[village].append(f)

print(f"\nUnannotated tiles by village:")
print(f"{'Village':<20} {'Count':>6}  {'Priority'}")
print("-" * 45)

total_remaining = 0
for village in sorted(unannotated.keys()):
    count = len(unannotated[village])
    total_remaining += count
    # Priority: villages with more tiles = more annotation bang for buck
    priority = "HIGH" if count > 50 else "MED" if count > 20 else "LOW"
    print(f"{village:<20} {count:>6}  {priority}")

print(f"\nTotal remaining: {total_remaining}")
print(f"Already done:    {len(annotated)}")
print(f"Target:          800+")
print(f"Still needed:    {max(0, 800 - len(annotated))}")

# Save unannotated list per village for easy Label Studio filtering
os.makedirs("outputs", exist_ok=True)
with open("outputs/unannotated_tiles.txt", "w") as f:
    for village in sorted(unannotated.keys()):
        f.write(f"\n=== {village} ({len(unannotated[village])} tiles) ===\n")
        for tile in unannotated[village]:
            f.write(tile + "\n")

print(f"\nDetailed list saved to: outputs/unannotated_tiles.txt")
print(f"\nAnnotation session plan:")
needed = max(0, 800 - len(annotated))
villages_sorted = sorted(unannotated.items(), key=lambda x: -len(x[1]))
covered = 0
for village, tiles in villages_sorted:
    if covered >= needed:
        break
    take = min(len(tiles), needed - covered)
    print(f"  {village}: annotate {take} tiles")
    covered += take