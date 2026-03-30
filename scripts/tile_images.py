import numpy as np
import os
import rasterio
from rasterio.windows import Window
import cv2

INPUT_FOLDER = "data/raw/training"
OUTPUT_FOLDER = "data/processed/tiles"
TILE_SIZE = 512

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# First delete old tiles
print("Cleaning old tiles...")
for f in os.listdir(OUTPUT_FOLDER):
    os.remove(os.path.join(OUTPUT_FOLDER, f))
print("Old tiles removed.")

for file in os.listdir(INPUT_FOLDER):
    if not file.endswith(".tif"):
        continue

    path = os.path.join(INPUT_FOLDER, file)
    print(f"\nProcessing: {file}")

    with rasterio.open(path) as src:
        width = src.width
        height = src.height
        print(f"  Size: {width} x {height}, Bands: {src.count}")

        # Step 1 — Read a sample to find global min/max
        # Sample every 50th row to estimate range quickly
        print("  Calculating global pixel range...")
        sample_window = Window(0, 0, min(width, 5000), min(height, 5000))
        sample = src.read([1, 2, 3], window=sample_window)

        global_min = float(sample.min())
        global_max = float(sample.max())
        print(f"  Global pixel range: {global_min} to {global_max}")

        if global_max == global_min:
            print("  Skipping — no data range found")
            continue

        count = 0

        # Step 2 — Tile using global normalization
        for y in range(0, height, TILE_SIZE):
            for x in range(0, width, TILE_SIZE):

                window = Window(x, y, TILE_SIZE, TILE_SIZE)
                tile = src.read([1, 2, 3], window=window)

                if tile.shape[1] != TILE_SIZE or tile.shape[2] != TILE_SIZE:
                    continue

                # Normalize using GLOBAL min/max (not per-tile)
                tile_norm = ((tile.astype(np.float32) - global_min) /
                             (global_max - global_min) * 255).clip(0, 255).astype(np.uint8)

                # Skip tiles that are mostly uniform (no content)
                if tile_norm.std() < 2.0:
                    continue

                # Convert C,H,W → H,W,C and save
                tile_rgb = np.transpose(tile_norm, (1, 2, 0))
                out_path = os.path.join(OUTPUT_FOLDER, f"{file}_{count}.png")
                cv2.imwrite(out_path, cv2.cvtColor(tile_rgb, cv2.COLOR_RGB2BGR))

                count += 1

        print(f"  Saved {count} tiles")

print("\nDone!")