# NeuroScan Brain Tumour Detection Backend

This backend module adapts the research notebook (`Brain_Tumour_Segmentation_(1).ipynb`) into a production-ready, modular computer vision processing engine.

## Key Changes from Research Notebook

| Research Notebook | NeuroScan Backend |
|---|---|
| Required `Brain_DATASET.zip` on disk | Accepts in-memory single MRI images (PIL Image, bytes, NumPy array) |
| Hardcoded dataset file loops | Instant single-slice inference suitable for interactive web apps |
| Interactive `plt.show()` inline plots | Structured output dataclass + base64 export + PIL images |
| Notebook script execution | Dual mode: Python module import **or** standalone FastAPI REST microservice |
| Edge-case crashes on empty contours | Robust fallbacks for healthy scans, small heads, and absent tumours |

---

## Architecture & Pipeline Steps

The algorithm executes the classical 10-step computer vision & watershed pipeline:

1. **Image Ingestion & Normalization**: Converts input (PNG, JPG, bytes) to standardized RGB NumPy array.
2. **Gaussian Smoothing**: $5 \times 5$ kernel with $\sigma=2$ suppresses high-frequency acquisition grain.
3. **First Enhancement**: Contrast boost via `cv.add(orig_rgb, cv.add(blur, -100))`.
4. **Median Denoising**: $5 \times 5$ median blur removes salt-and-pepper noise while preserving edges.
5. **Morphological Gradient & 2nd Enhancement**: Cross kernel gradient extraction blended with median denoised scan.
6. **Intracranial ROI Extraction**: Skull detection, contour extraction, and Euclidean distance transform exclusion of outer skull margin ($15\%$ default).
7. **Adaptive Thresholding**: Combined Otsu threshold with 85th percentile floor within the brain ROI, followed by morphological opening & closing.
8. **Marker-Based Watershed Setup**:
   - Distance transform on foreground.
   - Sure foreground ($> 50\%$ peak distance).
   - Dilated sure background.
   - Connected-component marker seeding.
9. **Topographic Watershed Segmentation**: Flooding intensity relief surface from seed markers.
10. **Candidate Extraction & Quantification**: Identification of largest candidate inside ROI, boundary contour tracing, pixel area summation, and perimeter calculation.

---

## Quickstart: Python Integration

```python
from PIL import Image
from Backend.pipeline import BrainTumourPipeline

# Initialize pipeline
pipeline = BrainTumourPipeline(roi_margin=0.15, percentile=85.0)

# Run analysis on any uploaded PIL Image or image bytes
image = Image.open("sample_mri.png")
result = pipeline.analyze(image)

print("Tumour Detected:", result["tumor_detected"])
print("Estimated Area:", result["region_area_px"], "pixels")
print("Perimeter:", result["perimeter_px"], "pixels")
print("Bounding Box:", result["bounding_box"])

# Access output images
overlay_img = result["overlay_image"]
mask_arr = result["mask_u8"]
```

---

## Quickstart: FastAPI REST Microservice

To run the backend as a standalone HTTP microservice:

```bash
python -m uvicorn Backend.api:app --host 0.0.0.0 --port 8000 --reload
```

- **Interactive Swagger Docs**: `http://localhost:8000/docs`
- **Health Check**: `GET http://localhost:8000/health`
- **Analyze Scan**: `POST http://localhost:8000/api/v1/analyze` (upload file as `file`)
