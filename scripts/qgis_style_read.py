import rasterio
import numpy as np

path = "data/raw/training/kutru_proper.tif"

with rasterio.open(path) as src:
    # Read center of image where village should be
    cx = src.width // 2
    cy = src.height // 2
    
    import rasterio.windows as w
    window = w.Window(cx - 256, cy - 256, 512, 512)
    
    data = src.read([1, 2, 3], window=window).astype(np.float32)
    
    print(f"Center patch values:")
    print(f"  Band1 — min:{data[0].min():.1f} max:{data[0].max():.1f} mean:{data[0].mean():.2f}")
    print(f"  Band2 — min:{data[1].min():.1f} max:{data[1].max():.1f} mean:{data[1].mean():.2f}")
    print(f"  Band3 — min:{data[2].min():.1f} max:{data[2].max():.1f} mean:{data[2].mean():.2f}")
    
    print(f"\nAll unique values in Band1: {np.unique(data[0])}")
    
    # Try rendering with aggressive stretch
    import cv2
    for band_idx in range(3):
        band = data[band_idx]
        stretched = ((band / band.max()) * 255).astype(np.uint8) if band.max() > 0 else band.astype(np.uint8)
        data[band_idx] = stretched
    
    tile_rgb = np.transpose(data.astype(np.uint8), (1, 2, 0))
    cv2.imwrite("outputs/center_patch.png", cv2.cvtColor(tile_rgb, cv2.COLOR_RGB2BGR))
    print("\nSaved center patch to outputs/center_patch.png")
    print("Open this file and tell me what you see")