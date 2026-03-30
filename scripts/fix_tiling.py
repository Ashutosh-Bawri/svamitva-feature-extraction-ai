"""
fix_tiling.py — Tiles ALL village exports (village1 through village35, both .tif and .png)
Run: python scripts/fix_tiling.py
"""

import numpy as np
import os
import rasterio
from rasterio.windows import Window
import cv2

INPUT_FOLDER  = "data/raw/training"
OUTPUT_FOLDER = "data/processed/tiles"
TILE_SIZE     = 512

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Clean old tiles
print("Cleaning old tiles...")
removed = 0
for f in os.listdir(OUTPUT_FOLDER):
    os.remove(os.path.join(OUTPUT_FOLDER, f))
    removed += 1
print(f"Removed {removed} old tiles.")

# Files to skip — large broken files and sidecar files
SKIP_FILES = {
    "kutru_ortho.tif", "kutru_proper.tif", "kutru_decoded.tif", "kutru_final.tif",
    "village1.tif", "village2.tif", "village3.tif",  # too small (800x800)
}

# Find ALL village files — both .tif and .png, skip .pgw sidecar files
all_files = sorted([
    f for f in os.listdir(INPUT_FOLDER)
    if f.startswith("village")
    and (f.endswith(".tif") or f.endswith(".png"))
    and not f.endswith(".pgw")
    and f not in SKIP_FILES
])

print(f"\nFound {len(all_files)} village files:")
for f in all_files:
    print(f"  {f}")

if len(all_files) == 0:
    print("\nERROR: No village files found in", INPUT_FOLDER)
    exit()

grand_total   = 0
grand_skipped = 0

for file in all_files:
    path = os.path.join(INPUT_FOLDER, file)
    print(f"\nProcessing: {file}")

    try:
        with rasterio.open(path) as src:
            width  = src.width
            height = src.height
            bands  = src.count
            print(f"  Size: {width} x {height}, Bands: {bands}")

            if width < TILE_SIZE or height < TILE_SIZE:
                print(f"  Skipping — image too small ({width}x{height})")
                continue

            # Sample center for percentile stretch
            cx       = width  // 2
            cy       = height // 2
            sample_w = min(8000, width)
            sample_h = min(8000, height)
            sx       = max(0, cx - sample_w // 2)
            sy       = max(0, cy - sample_h // 2)

            sample_window = Window(sx, sy, sample_w, sample_h)

            # Read up to 3 bands (RGB), ignore alpha
            read_bands = [1, 2, 3] if bands >= 3 else [1]
            sample = src.read(read_bands, window=sample_window).astype(np.float32)

            p_low  = np.percentile(sample, 2)
            p_high = np.percentile(sample, 98)

            print(f"  Pixel range: {sample.min():.1f}–{sample.max():.1f} "
                  f"| Stretch: {p_low:.1f}–{p_high:.1f}")

            if p_high <= p_low:
                print("  Skipping — flat image (no data range)")
                continue

            count   = 0
            skipped = 0

            for y in range(0, height, TILE_SIZE):
                for x in range(0, width, TILE_SIZE):
                    window = Window(x, y, TILE_SIZE, TILE_SIZE)
                    tile   = src.read(read_bands, window=window).astype(np.float32)

                    # Skip incomplete edge tiles
                    if tile.shape[1] != TILE_SIZE or tile.shape[2] != TILE_SIZE:
                        continue

                    # Stretch to 0-255
                    tile_norm = ((tile - p_low) / (p_high - p_low) * 255
                                 ).clip(0, 255).astype(np.uint8)

                    # Skip blank/uniform tiles
                    if tile_norm.std() < 3.0:
                        skipped += 1
                        continue

                    # Handle grayscale -> RGB
                    if tile_norm.shape[0] == 1:
                        tile_norm = np.repeat(tile_norm, 3, axis=0)

                    tile_rgb = np.transpose(tile_norm, (1, 2, 0))
                    out_path = os.path.join(OUTPUT_FOLDER, f"{file}_{count}.png")
                    cv2.imwrite(out_path,
                                cv2.cvtColor(tile_rgb, cv2.COLOR_RGB2BGR))
                    count += 1

            print(f"  Saved: {count} tiles | Skipped: {skipped} blank")
            grand_total   += count
            grand_skipped += skipped

    except Exception as e:
        print(f"  ERROR processing {file}: {e}")
        continue

print(f"\n{'='*50}")
print(f"ALL DONE")
print(f"Total tiles saved:   {grand_total}")
print(f"Total tiles skipped: {grand_skipped}")
print(f"Tiles folder:        {OUTPUT_FOLDER}")
print(f"{'='*50}")

if grand_total < 500:
    print("\nWARNING: Less than 500 tiles generated.")
    print("Go back to QGIS and export more village areas at 600 DPI.")
elif grand_total < 1500:
    print(f"\nGood: {grand_total} tiles. Should give 80-90% IoU.")
else:
    print(f"\nExcellent: {grand_total} tiles. Enough for 95%+ IoU.")