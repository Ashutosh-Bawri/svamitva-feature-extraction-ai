import rasterio
import os

INPUT_FOLDER = "data/raw/training"
OUTPUT_FOLDER = "data/processed/tiles"

tile_size = 512

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

for file in os.listdir(INPUT_FOLDER):

    if file.endswith(".tif"):

        path = os.path.join(INPUT_FOLDER, file)

        with rasterio.open(path) as src:

            width = src.width
            height = src.height

            count = 0

            for y in range(0, height, tile_size):
                for x in range(0, width, tile_size):

                    window = rasterio.windows.Window(x, y, tile_size, tile_size)

                    tile = src.read(window=window)

                    if tile.shape[1] == tile_size and tile.shape[2] == tile_size:

                        output = os.path.join(OUTPUT_FOLDER, f"{file}_{count}.tif")

                        with rasterio.open(
                            output,
                            'w',
                            driver='GTiff',
                            height=tile_size,
                            width=tile_size,
                            count=3,
                            dtype=tile.dtype
                        ) as dst:

                            dst.write(tile)

                        count += 1