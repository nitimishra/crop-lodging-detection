# 🌾 Crop Lodging Detection Portal

An interactive, remote sensing, and machine learning dashboard designed to quantify and detect crop lodging damage using UAV (drone) multispectral orthomosaics. This portal empowers farmers, insurers, and agronomists to assess field health, analyze vegetation distributions, and estimate insurance coverage under the **Pradhan Mantri Fasal Bima Yojana (PMFBY)** guidelines.

---

## 🚀 Key Features

* **🔐 Gated Authentication:** Fully secure login and registration flow with central SQLite database management (`database/users.db`) and session persistence across browser refreshes.
* **🌐 Bilingual Localisation:** Complete translation dictionary supporting instantaneous language toggle between **English** and **Hindi (हिंदी)**.
* **🗺️ Interactive Lodging Mapping:** Dynamic GIS map layers separating lodged crop regions (Red) from healthy/non-lodged crops (Green) over a 5-meter grid.
* **📊 Spectral Analytics:** Detailed value distributions and descriptive statistics (mean, median, min, max) for 26 computed Vegetation Indices (NDVI, NDRE, LCI, GNDVI, etc.).
* **🧮 PMFBY Premium Calculator:** Localized insurance premium estimation tool utilizing real PMFBY rate structures and dynamic dependent dropdown lists for States and Districts.
* **⚡ Production-Ready CSS Overrides:** Clean, custom UI with custom-branded color palettes, styled container cards, hidden utility elements, and responsive layout structures.

---

## 🛠️ Tech Stack & Dependencies

* **Frontend:** Streamlit, Folium (`streamlit-folium`), Pandas, NumPy, Matplotlib
* **Backend:** FastAPI, Uvicorn, Rasterio, GeoPandas, Shapely, Rasterstats, Joblib
* **Machine Learning Model:** Support Vector Machine (Linear SVC) trained on multispectral band values and vegetation indices
* **Database:** SQLite3

---

## 📂 Project Structure

```text
crop-lodging-webapp/
├── backend/
│   ├── main.py             # FastAPI backend (API endpoints & ML predictions)
│   └── vi_formulas.py      # Core functions for computing 26 Vegetation Indices
├── database/
│   └── users.db            # Centralized database for user profiles (automatically generated)
├── frontend/
│   └── app.py              # Streamlit frontend dashboard (UI components, maps, & CSS)
├── Models/
│   ├── final_model.pkl     # Trained Support Vector Machine model binary
│   ├── final_scaler.pkl    # Data scaler object
│   ├── feature_names.pkl   # List of features used in the model
│   └── metadata.pkl        # Model configuration metadata
├── Dockerfile              # Docker recipe for Hugging Face Spaces deployment
├── .gitignore              # Tells Git to ignore database, venv, and large imagery files
├── requirements.txt        # Exact python packages and dependencies
└── README.md               # Project documentation
```

---

## ⚙️ Local Installation & Running

### Prerequisites
Make sure you have Python 3.11+ installed on your computer.

### Step 1: Clone the Repository
```bash
git clone <your-github-repo-url>
cd crop-lodging-webapp
```

### Step 2: Set Up Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows (PowerShell):
venv\Scripts\Activate.ps1
# On Mac/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Run the Backend API
Start the FastAPI server on port `8000`:
```bash
# Run from the root directory
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

### Step 5: Run the Streamlit Frontend
Start the Streamlit dashboard in a separate terminal:
```bash
streamlit run frontend/app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser to access the portal.

---

## ☁️ Deployment Guide

### Option 1: Frontend on Streamlit Cloud & Backend on Render
1. Push your code to GitHub.
2. Deploy backend on **Render** (New -> Web Service).
   * **Build Command:** `pip install -r requirements.txt`
   * **Start Command:** `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
3. Deploy frontend on **Streamlit Community Cloud** (share.streamlit.io).
4. Add the Render API URL as a secret in Streamlit Cloud Settings:
   ```toml
   BACKEND_URL = "https://your-backend-app.onrender.com"
   ```

### Option 2: 100% Free Tunneling with ngrok (Recommended for Heavy Datasets)
Render's free tier is limited to 512 MB of RAM, which can crash on processing large `.tif` imagery. To run the calculations on your local laptop's memory for free:
1. Start your local FastAPI backend on port `8000`.
2. Run ngrok tunnel command:
   ```bash
   ngrok http 8000
   ```
3. Copy the generated public HTTPS URL (e.g., `https://abcd-1234.ngrok-free.app`).
4. Update the **Secrets** settings on your Streamlit Cloud:
   ```toml
   BACKEND_URL = "https://abcd-1234.ngrok-free.app"
   ```

---

## 🔒 Security & Best Practices
* **Database Exclusions:** The central database folder `database/` is listed in the `.gitignore` file to prevent user profile credentials from leaking to GitHub.
* **Large Files Exclusions:** Aerial images (`.tif`/`.tiff` files) are automatically ignored by Git to avoid repository bloat and ensure fast pushing.
