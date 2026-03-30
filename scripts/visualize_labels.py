import cv2
import os
import matplotlib.pyplot as plt

TILE_DIR = "data/processed/tiles"
MASK_DIR = "data/processed/masks"

# Pick tiles that had both buildings and roads detected
interesting = [
    "village1.tif_0",
    "village1_hd.tif_0",
    "village2_hd.tif_12",
    "village3_hd.tif_21",
    "village4_hd.tif_2",
    "village6_hd.tif_39",
    "village8_hd.tif_102",
    "village9_hd.tif_41",
]

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()

for i, name in enumerate(interesting):
    tile_path = os.path.join(TILE_DIR, name + ".png")
    overlay_path = os.path.join(MASK_DIR, name + "_overlay.png")

    if os.path.exists(overlay_path):
        img = cv2.imread(overlay_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axes[i].imshow(img)
        axes[i].set_title(name[-20:], fontsize=7)
    else:
        axes[i].set_title(f"{name}\n(not found)", fontsize=7)
    axes[i].axis("off")

# Add legend
fig.text(0.5, 0.02,
    "Red = Buildings | Green = Roads | Blue = Water | No color = Background",
    ha="center", fontsize=11)

plt.tight_layout()
os.makedirs("outputs", exist_ok=True)
plt.savefig("outputs/label_preview.png", dpi=150)
plt.show()
print("Saved to outputs/label_preview.png")