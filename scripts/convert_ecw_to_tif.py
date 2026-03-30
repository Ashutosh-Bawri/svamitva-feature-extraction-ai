from osgeo import gdal

input_file = r"C:\Users\ashut\OneDrive\Desktop\svamitva_ai_project\data\raw\training\KUTRU_451189_AAKLANKA_451163_ORTHO_3857.ecw"
output_file = r"C:\Users\ashut\OneDrive\Desktop\svamitva_ai_project\data\raw\training\kutru_ortho.tif"

dataset = gdal.Open(input_file)

gdal.Translate(output_file, dataset, format="GTiff")

print("Conversion completed successfully!")