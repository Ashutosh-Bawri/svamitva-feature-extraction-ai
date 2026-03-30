# AI-Based Feature Extraction from Drone Orthophotos

End-to-end GeoAI pipeline for extracting buildings, roads, and water bodies from drone imagery.


## Overview
This project develops a GeoAI pipeline to extract key features such as buildings, roads, and water bodies from SVAMITVA drone imagery using deep learning.

---

## Problem Statement
Manual extraction of geospatial features from drone orthophotos is time-consuming and inefficient. This project automates the process using AI.

---

## Methodology
- Converted ECW to GeoTIFF
- Tiled images into 512×512 patches (17,488 tiles)
- Trained DeepLabV3+ model for segmentation
- Applied post-processing for feature extraction
- Generated GIS-compatible GeoJSON output

---

## Results
- Buildings detected: 4,495
- Roads detected: 6,859
- Water bodies detected: 7,589
- Total features: 18,943

---

## Output Files
- `eval_results.png` → segmentation results
- `extracted_features.geojson` → GIS-ready extracted features

---

## Technologies Used
- Python
- PyTorch
- OpenCV
- Rasterio
- QGIS

---
## How to Run

```bash
python scripts/predict_and_visualize.py
python scripts/predict_geojson.py
```

---

## Model Download

Due to file size limits, the trained model is available here:

[Download Model](https://drive.google.com/file/d/1YIeAiHzzsoInVypI5TeCD_E2fheapa9s/view?usp=sharing)

---

## Sample Data

A few sample tiles are provided in the `sample_data/` folder for testing.

---


## Results

![Output](outputs/eval_results.png)

---
