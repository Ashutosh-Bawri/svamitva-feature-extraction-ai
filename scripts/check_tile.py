import rasterio
import numpy as np
import matplotlib.pyplot as plt

# Check one tile
path = "data/processed/tiles/kutru_ortho.tif_0.png"

with rasterio.open(path) as src:
    print("Width:", src.width)
    print("Height:", src.height)
    print("Bands:", src.count)
    print("Dtype:", src.dtypes)
    
    image = src.read()
    print("Min pixel value:", image.min())
    print("Max pixel value:", image.max())
    print("Mean pixel value:", image.mean())

# Also check the original source image
path2 = "data/raw/training/kutru_ortho.tif"

with rasterio.open(path2) as src:
    print("\n--- Original image ---")
    print("Width:", src.width)
    print("Height:", src.height)
    print("Bands:", src.count)
    print("Dtype:", src.dtypes)
    
    # Read just a small patch
    window = rasterio.windows.Window(0, 0, 512, 512)
    patch = src.read(window=window)
    print("Min pixel value:", patch.min())
    print("Max pixel value:", patch.max())
    print("Mean pixel value:", patch.mean())