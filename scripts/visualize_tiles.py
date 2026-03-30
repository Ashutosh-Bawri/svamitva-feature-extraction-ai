import cv2
import os
import matplotlib.pyplot as plt

INPUT_DIR = "data/processed/tiles"

# Auto-pick all PNG files
all_files = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith(".png")])

print(f"Total tiles found: {len(all_files)}")

if len(all_files) == 0:
    print("No tiles found!")
else:
    # Pick 8 spread across dataset
    step = max(1, len(all_files) // 8)
    tiles_to_check = [all_files[i] for i in range(0, len(all_files), step)][:8]

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for i, filename in enumerate(tiles_to_check):
        path = os.path.join(INPUT_DIR, filename)
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axes[i].imshow(img)
        axes[i].set_title(filename[:30], fontsize=7)
        axes[i].axis("off")

    # Hide unused subplots
    for i in range(len(tiles_to_check), 8):
        axes[i].axis("off")

    plt.tight_layout()
    os.makedirs("outputs", exist_ok=True)
    plt.savefig("outputs/tile_preview.png", dpi=150)
    plt.show()
    print("Saved to outputs/tile_preview.png")