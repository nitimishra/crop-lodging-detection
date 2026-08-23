"""
vi_formulas.py
================
Calculates all 26 vegetation indices from the 5 raw bands (Blue, Green, Red,
RedEdge, NIR), pixel-by-pixel, using numpy arrays.

These formulas are the EXACT expressions used in the original QGIS Raster
Calculator project (verified against the saved VI formula document) — both
the 7march and 12march ("march_ortho") QGIS projects use the identical
mathematical formula, just with different input raster/band names, so a
single formula set here correctly reproduces both.

⚠️ ONE EXCEPTION: "Ndvi1" had no separate documented formula (only "NDVI"
was documented). It is treated here as identical to NDVI — if this is
actually a different index in your project, update it below.
"""

import numpy as np

EPSILON = 1e-8  # tiny value added to denominators to avoid divide-by-zero errors
                 # (does not meaningfully change results, just prevents crashes/inf)


def compute_all_vis(blue: np.ndarray, green: np.ndarray, red: np.ndarray,
                     rededge: np.ndarray, nir: np.ndarray) -> dict:
    """
    Takes the 5 raw band arrays and returns a dict of {vi_name: array}
    for all 26 vegetation indices, using the SAME NAMES your Kaggle
    pipeline extracted (must exactly match feature_names.pkl).
    """
    vis = {}

    vis["ARVI"] = (nir - (2 * red - blue)) / (nir + (2 * red - blue) + EPSILON)

    vis["CIGreen"] = (nir / (green + EPSILON)) - 1

    vis["CIRedEdge"] = (nir / (rededge + EPSILON)) - 1

    vis["EVI"] = 2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1 + EPSILON)

    vis["EXG"] = 2 * green - red - blue

    vis["CLI"] = (rededge / (red + EPSILON)) - 1

    vis["GNDVI"] = (nir - green) / (nir + green + EPSILON)

    vis["GRVI"] = nir / (green + EPSILON)

    vis["IPVI"] = nir / (nir + red + EPSILON)

    vis["MCARI"] = ((rededge - red) - 0.2 * (rededge - green)) * (rededge / (red + EPSILON))

    vis["MSAVI2"] = (2 * nir + 1 - np.sqrt(
        np.clip((2 * nir + 1) ** 2 - 8 * (nir - red), 0, None)
    )) / 2

    vis["MTVI2"] = (1.5 * (1.2 * (nir - green) - 2.5 * (red - green))) / (np.sqrt(np.clip(
        (2 * nir + 1) ** 2 - (6 * nir - 5 * np.sqrt(np.clip(red, 0, None))) - 0.5, 0, None
    )) + EPSILON)

    vis["NDRE"] = (nir - rededge) / (nir + rededge + EPSILON)

    vis["NDVI"] = (nir - red) / (nir + red + EPSILON)

    vis["NDWI"] = (green - nir) / (green + nir + EPSILON)

    vis["PSRI"] = (red - blue) / (rededge + EPSILON)

    vis["RTVI"] = 100 * (nir - rededge) - 10 * (nir - green)  # was "RTVICore" in the raw files

    L = 0.5
    vis["SAVI"] = ((nir - red) / (nir + red + L + EPSILON)) * (1 + L)

    vis["TVI"] = 0.5 * (120 * (nir - green) - 200 * (red - green))

    vis["VARI"] = (green - red) / (green + red - blue + EPSILON)

    a = 0.1
    vis["WDRVI"] = (a * nir - red) / (a * nir + red + EPSILON)

    vis["NGRVI"] = (green - red) / (green + red + EPSILON)

    vis["MGRVI"] = (green ** 2 - red ** 2) / (green ** 2 + red ** 2 + EPSILON)

    vis["NGBVI"] = (green - blue) / (green + blue + EPSILON)

    vis["VDVI"] = (2 * green - red - blue) / (2 * green + red + blue + EPSILON)

    # ⚠️ No separate documented formula for "Ndvi1" — assumed identical to NDVI.
    vis["Ndvi1"] = vis["NDVI"]

    return vis


# The 5 raw bands, kept as-is (no formula needed, just the reflectance values)
BAND_NAMES = ["BLUE", "GREEN", "RED", "REDEDGE", "NIR"]