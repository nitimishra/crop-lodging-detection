"""
main.py — FastAPI backend for Crop Lodging Detection

Workflow:
1. Receives an uploaded multispectral orthomosaic (.tif, 5 bands)
2. Reprojects to the training CRS if needed
3. Computes all 26 VI rasters from the 5 raw bands
4. Builds a grid over the orthomosaic extent (same cell size as training)
5. Runs zonal statistics (mean) per grid cell, for all 31 features
6. Scales features and predicts lodged / non-lodged per cell
7. Returns a summary (areas, percentages) + GeoJSON for map display

Run with:  uvicorn main:app --reload --port 8000   (from the backend/ folder)
"""

import os
import sys
import tempfile
import re

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from shapely.geometry import box
from rasterstats import zonal_stats
import joblib

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

sys.path.append(os.path.dirname(__file__))
from vi_formulas import compute_all_vis

# --- CONFIG (⚠️ edit these to match your training setup) ---
TARGET_CRS = "EPSG:32643"     # same CRS used when building the training grid in QGIS
GRID_CELL_SIZE = 5.0           # meters — ⚠️ must match your QGIS "Create Grid" cell size

# --- Load saved model artifacts ---
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "Models")
model = joblib.load(os.path.join(MODELS_DIR, "final_model.pkl"))
scaler = joblib.load(os.path.join(MODELS_DIR, "final_scaler.pkl"))
feature_names = joblib.load(os.path.join(MODELS_DIR, "feature_names.pkl"))
label_mapping = joblib.load(os.path.join(MODELS_DIR, "label_mapping.pkl"))
metadata = joblib.load(os.path.join(MODELS_DIR, "metadata.pkl"))
INV_LABEL_MAP = {v: k for k, v in label_mapping.items()}

app = FastAPI(title="Crop Lodging Detection API")

# Allow the Streamlit frontend (running on a different port) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "status": "Crop Lodging Detection API is running",
        "model": metadata.get("best_model_name"),
        "num_features": metadata.get("num_features"),
    }


def match_files_to_bands(files: list[UploadFile]) -> dict:
    """
    Identifies which UploadFile corresponds to which of the 5 bands:
    BLUE, GREEN, RED, REDEDGE, NIR.
    Uses scoring based on filename patterns.
    """
    matched = {}
    scores = []
    for f in files:
        name = f.filename.lower()
        f_scores = {
            "BLUE": 0,
            "GREEN": 0,
            "RED": 0,
            "REDEDGE": 0,
            "NIR": 0
        }
        
        # RedEdge check (do first to prevent overlapping with red)
        if any(k in name for k in ["rededge", "red_edge", "red-edge", "re.tif", "_re", "_4.tif"]):
            f_scores["REDEDGE"] += 10
        elif "edge" in name:
            f_scores["REDEDGE"] += 8
            
        # NIR check
        if any(k in name for k in ["nir", "near", "n.tif", "_n", "b5", "band5", "_5.tif"]):
            f_scores["NIR"] += 10
            
        # Blue check
        if any(k in name for k in ["blue", "b.tif", "_b", "b1", "band1", "_1.tif"]):
            f_scores["BLUE"] += 10
            
        # Green check
        if any(k in name for k in ["green", "g.tif", "_g", "b2", "band2", "_2.tif"]):
            f_scores["GREEN"] += 10
            
        # Red check (only if not RedEdge keywords)
        if "red" in name and not any(k in name for k in ["rededge", "red_edge", "red-edge", "edge"]):
            f_scores["RED"] += 10
        if any(k in name for k in ["r.tif", "_r", "b3", "band3", "_3.tif"]) and not any(k in name for k in ["rededge", "red_edge", "red-edge", "edge"]):
            f_scores["RED"] += 10
            
        # Suffix and B-index patterns (e.g. _1.tif, b1, band1)
        match_num = re.search(r'[-_]([1-5])\.(?:tif|tiff)$', name)
        if match_num:
            idx = int(match_num.group(1))
            mapping = {1: "BLUE", 2: "GREEN", 3: "RED", 4: "REDEDGE", 5: "NIR"}
            f_scores[mapping[idx]] += 15
            
        match_b = re.search(r'\bb([1-5])\b', name)
        if match_b:
            idx = int(match_b.group(1))
            mapping = {1: "BLUE", 2: "GREEN", 3: "RED", 4: "REDEDGE", 5: "NIR"}
            f_scores[mapping[idx]] += 15
            
        scores.append((f, f_scores))
        
    remaining_files = list(files)
    for band in ["REDEDGE", "NIR", "BLUE", "GREEN", "RED"]:
        best_file = None
        best_score = -1
        for f, f_scores in scores:
            if f in remaining_files and f_scores[band] > best_score and f_scores[band] > 0:
                best_score = f_scores[band]
                best_file = f
        if best_file:
            matched[band] = best_file
            remaining_files.remove(best_file)
            
    # Alphabetical fallback if exactly 5 files and not all are matched
    if len(matched) < 5 and len(files) == 5:
        sorted_files = sorted(files, key=lambda f: f.filename.lower())
        matched = {
            "BLUE": sorted_files[0],
            "GREEN": sorted_files[1],
            "RED": sorted_files[2],
            "REDEDGE": sorted_files[3],
            "NIR": sorted_files[4]
        }
        
    return matched


@app.post("/predict")
async def predict(files: list[UploadFile] = File(...)):
    if not files:
        return JSONResponse(status_code=400, content={"error": "No files uploaded"})

    temp_paths = {}
    try:
        if len(files) == 1:
            file = files[0]
            with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as tmp:
                tmp.write(await file.read())
                tmp_path = tmp.name
            temp_paths["single"] = tmp_path

            with rasterio.open(tmp_path) as src:
                if src.count < 5:
                    return JSONResponse(status_code=400,
                                         content={"error": f"Expected at least 5 bands in a single file upload, found {src.count}"})

                if str(src.crs) != TARGET_CRS:
                    transform, width, height = calculate_default_transform(
                        src.crs, TARGET_CRS, src.width, src.height, *src.bounds)
                    data = np.zeros((src.count, height, width), dtype=np.float32)
                    for i in range(1, src.count + 1):
                        reproject(
                            source=rasterio.band(src, i),
                            destination=data[i - 1],
                            src_transform=src.transform, src_crs=src.crs,
                            dst_transform=transform, dst_crs=TARGET_CRS,
                            resampling=Resampling.bilinear,
                        )
                    affine = transform
                    bounds = rasterio.transform.array_bounds(height, width, transform)
                else:
                    data = src.read().astype(np.float32)
                    affine = src.transform
                    bounds = src.bounds

                blue, green, red, rededge, nir = data[0], data[1], data[2], data[3], data[4]
        else:
            matched = match_files_to_bands(files)
            missing = [b for b in ["BLUE", "GREEN", "RED", "REDEDGE", "NIR"] if b not in matched]
            if missing:
                file_names_str = ", ".join([f.filename for f in files])
                matched_str = ", ".join([f"{b}: {f.filename}" for b, f in matched.items()])
                return JSONResponse(status_code=400, content={
                    "error": f"Could not identify all 5 bands from filenames. "
                             f"Missing: {', '.join(missing)}. "
                             f"Uploaded files: [{file_names_str}]. "
                             f"Identified: {{{matched_str}}}. "
                             f"Please rename files to include band keywords (blue, green, red, rededge, nir) or suffix numbers 1 to 5."
                })

            for band_name, upload_file in matched.items():
                with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as tmp:
                    tmp.write(await upload_file.read())
                    temp_paths[band_name] = tmp.name

            anchor_band = "BLUE"
            with rasterio.open(temp_paths[anchor_band]) as anchor_src:
                anchor_crs = anchor_src.crs
                if str(anchor_crs) != TARGET_CRS:
                    transform, width, height = calculate_default_transform(
                        anchor_crs, TARGET_CRS, anchor_src.width, anchor_src.height, *anchor_src.bounds)
                    dst_crs = TARGET_CRS
                else:
                    transform = anchor_src.transform
                    width = anchor_src.width
                    height = anchor_src.height
                    dst_crs = TARGET_CRS

                band_data = {}
                for band_name in ["BLUE", "GREEN", "RED", "REDEDGE", "NIR"]:
                    with rasterio.open(temp_paths[band_name]) as src:
                        data_arr = np.zeros((height, width), dtype=np.float32)
                        reproject(
                            source=rasterio.band(src, 1),
                            destination=data_arr,
                            src_transform=src.transform,
                            src_crs=src.crs,
                            dst_transform=transform,
                            dst_crs=dst_crs,
                            resampling=Resampling.bilinear,
                        )
                        band_data[band_name] = data_arr

                affine = transform
                bounds = rasterio.transform.array_bounds(height, width, transform)

                blue = band_data["BLUE"]
                green = band_data["GREEN"]
                red = band_data["RED"]
                rededge = band_data["REDEDGE"]
                nir = band_data["NIR"]

        # --- Compute all 26 VI rasters from the 5 raw bands ---
        vi_arrays = compute_all_vis(blue, green, red, rededge, nir)
        band_arrays = {"BLUE": blue, "GREEN": green, "RED": red, "REDEDGE": rededge, "NIR": nir}
        all_arrays = {**band_arrays, **vi_arrays}

        # --- Build a grid over the raster extent ---
        minx, miny, maxx, maxy = bounds
        cell = GRID_CELL_SIZE
        polygons = []
        x = minx
        while x < maxx:
            y = miny
            while y < maxy:
                polygons.append(box(x, y, x + cell, y + cell))
                y += cell
            x += cell

        grid = gpd.GeoDataFrame({"geometry": polygons}, crs=TARGET_CRS)
        grid["cell_id"] = range(len(grid))

        # --- Zonal statistics (mean) per cell, for every required feature ---
        feature_matrix = {}
        for name in feature_names:
            arr = all_arrays.get(name)
            if arr is None:
                return JSONResponse(status_code=500,
                                     content={"error": f"Missing VI computation for feature '{name}'. "
                                                        f"Check vi_formulas.py."})
            stats = zonal_stats(grid, arr, affine=affine, stats=["mean"], nodata=np.nan)
            feature_matrix[name] = [s["mean"] for s in stats]

        X_df = pd.DataFrame(feature_matrix)[feature_names]  # enforce exact training column order
        valid_mask = X_df.notna().all(axis=1)
        X_valid = X_df[valid_mask].values
        grid_valid = grid[valid_mask].reset_index(drop=True)

        if len(X_valid) == 0:
            return JSONResponse(status_code=400,
                                 content={"error": "No valid grid cells found — check orthomosaic "
                                                    "coverage, CRS, and band order."})

        # --- Scale + predict ---
        X_scaled = scaler.transform(X_valid)
        preds = model.predict(X_scaled)
        if hasattr(model, "predict_proba"):
            confidences = model.predict_proba(X_scaled).max(axis=1)
        else:
            confidences = np.full(len(preds), np.nan)

        grid_valid["prediction"] = [str(INV_LABEL_MAP[int(p)]) for p in preds]
        grid_valid["confidence"] = [float(c) for c in confidences]

        # Add all 31 feature values to grid_valid properties so they are in the geojson
        for name in feature_names:
            grid_valid[name] = X_df[valid_mask][name].values

        cell_area = GRID_CELL_SIZE * GRID_CELL_SIZE
        lodged_count = int((grid_valid["prediction"] == "lodged").sum())
        non_lodged_count = int((grid_valid["prediction"] == "non-lodged").sum())
        total_count = lodged_count + non_lodged_count

        summary = {
            "total_cells": total_count,
            "lodged_cells": lodged_count,
            "non_lodged_cells": non_lodged_count,
            "lodged_area_sqm": round(lodged_count * cell_area, 2),
            "non_lodged_area_sqm": round(non_lodged_count * cell_area, 2),
            "total_area_sqm": round(total_count * cell_area, 2),
            "lodged_percentage": round(100 * lodged_count / total_count, 2) if total_count else 0,
        }

        # Calculate statistics for indices and bands
        stats_df = X_df[valid_mask].describe().T
        stats_df = stats_df.round(4)
        feature_stats = stats_df.reset_index().rename(columns={"index": "feature"}).to_dict(orient="records")

        geojson = grid_valid.to_crs("EPSG:4326").__geo_interface__

        return {"summary": summary, "geojson": geojson, "feature_stats": feature_stats}

    finally:
        for path in temp_paths.values():
            try:
                os.remove(path)
            except Exception:
                pass