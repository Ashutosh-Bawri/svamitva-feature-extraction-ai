import rasterio
import numpy as np

path = "data/raw/training/village1.tif"
with rasterio.open(path) as src:
    print("Driver:", src.driver)
    print("CRS:", src.crs)
    print("Transform:", src.transform)
    print("Bands:", src.count)
    print("Dtype:", src.dtypes)
    print("Width:", src.width, "Height:", src.height)
    print("Tags:", src.tags())
    
    # Read band by band from center of image
    cx = src.width // 2
    cy = src.height // 2
    
    import rasterio.windows as w
    window = w.Window(cx, cy, 512, 512)
    
    for b in range(1, src.count + 1):
        band = src.read(b, window=window)
        print(f"\nBand {b}:")
        print(f"  min={band.min()}, max={band.max()}, mean={band.mean():.4f}")
        print(f"  unique values (first 20): {np.unique(band)[:20]}")
        # Count non-zero pixels
        nonzero = np.count_nonzero(band)
        print(f"  non-zero pixels: {nonzero} / {512*512} ({100*nonzero/(512*512):.1f}%)")