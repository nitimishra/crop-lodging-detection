"""
app.py — Streamlit frontend for Crop Lodging Detection

Run with:  streamlit run app.py   (from the frontend/ folder)
Make sure the FastAPI backend is running first:
  uvicorn main:app --reload --port 8000   (from the backend/ folder)
"""

import streamlit as st
import requests
import folium
from streamlit_folium import st_folium
import pandas as pd
import numpy as np
import branca.colormap as cm
import sqlite3

# --- SQLite Database Helper Functions ---
import os
import shutil

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
# Point database to central database/users.db in the project root folder
DB_PATH = os.path.join(PROJECT_ROOT, "database", "users.db")
DB_DIR = os.path.dirname(DB_PATH)

os.makedirs(DB_DIR, exist_ok=True)

# Auto-migrate/merge user database if files exist elsewhere
old_db_in_frontend = os.path.join(CURRENT_DIR, "users.db")
old_db_in_frontend_sub = os.path.join(CURRENT_DIR, "database", "users.db")
old_db_in_backend_sub = os.path.join(PROJECT_ROOT, "backend", "database", "users.db")
old_db_in_root = os.path.join(PROJECT_ROOT, "users.db")

# If the central database doesn't exist, try to copy from any existing backup database
if not os.path.exists(DB_PATH):
    for path in [old_db_in_backend_sub, old_db_in_frontend_sub, old_db_in_frontend, old_db_in_root]:
        if os.path.exists(path):
            try:
                shutil.copy(path, DB_PATH)
                break
            except Exception:
                pass

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password TEXT
        )
    """)
    conn.commit()
    conn.close()

def register_user(username, password) -> bool:
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        return False

def validate_user(username, password) -> bool:
    print(f"DEBUG: validate_user called with username={repr(username)}, password={repr(password)}")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE username = ? AND password = ?", (username, password))
    user = cursor.fetchone()
    conn.close()
    print(f"DEBUG: validate_user query result={repr(user)}")
    return user is not None

# Initialize user database
init_db()

def fix_geojson_coordinates(geojson: dict) -> dict:
    """
    Checks if GeoJSON coordinates are in [lat, lon] order instead of [lon, lat]
    and swaps them. Also, if coordinates are in meters (UTM) due to a reprojection 
    issue in the backend, it dynamically translates them to Bhopal, India degrees 
    to ensure the map renders correctly.
    """
    if not geojson or "features" not in geojson or not geojson["features"]:
        return geojson
        
    first_feature = geojson["features"][0]
    if "geometry" not in first_feature or "coordinates" not in first_feature["geometry"]:
        return geojson
        
    coords_list = first_feature["geometry"]["coordinates"][0]
    if not coords_list:
        return geojson
        
    first_coord = coords_list[0]
    
    # Case 1: Coordinates are in meters (UTM) because values are too large
    if abs(first_coord[1]) > 90 or abs(first_coord[0]) > 180:
        # Bhopal UTM Zone 43N bounds: left=745644.69, bottom=2580306.08
        # Bhopal degrees center: Lon=77.40216, Lat=23.31404
        for f in geojson["features"]:
            if "geometry" in f and "coordinates" in f["geometry"]:
                coords = f["geometry"]["coordinates"][0]
                new_coords = []
                for c in coords:
                    # Map UTM coordinates to degrees centered at Bhopal
                    lon_d = 77.40216 + (c[0] - 745644.69) * 0.0000096
                    lat_d = 23.31404 + (c[1] - 2580306.08) * 0.0000090
                    new_coords.append([lon_d, lat_d])
                f["geometry"]["coordinates"][0] = new_coords
        return geojson
        
    # Case 2: Coordinates are in degrees, but swapped to [latitude, longitude]
    # For India, longitude is ~77 (larger) and latitude is ~23 (smaller)
    if first_coord[0] < first_coord[1]:
        for f in geojson["features"]:
            if "geometry" in f and "coordinates" in f["geometry"]:
                f["geometry"]["coordinates"][0] = [[c[1], c[0]] for c in f["geometry"]["coordinates"][0]]
                   
    return geojson

# --- Page config ---
st.set_page_config(
    page_title="Crop Lodging Portal",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="collapsed", # Collapsed by default for clean home view
)
BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")

# --- Complete Translation Dictionary ---
TRANSLATIONS = {
    "English": {
        "portal_title": "Crop Lodging Detection Portal",
        "portal_subtitle": "Empowering Farmers, Insurers, & Agronomists with UAV Remote Sensing & Analytics",
        "lang_select": "Choose Language / भाषा चुनें",
        "login_header": "🔐 Portal Login",
        "login_sub": "Please enter your mobile number and password to access the portal.",
        "username_label": "Mobile / Username *",
        "username_placeholder": "Enter Mobile Number",
        "password_label": "Password *",
        "password_placeholder": "Enter Password",
        "login_btn": "🔑 Login",
        "login_err": "Please enter username and password.",
        "logout_btn": "❌ Logout",
        "role_label": "Access Level: Farmer (Bhopal Region)",
        "sidebar_info_title": "🌾 Quick Info",
        "sidebar_info_desc": "This platform analyzes UAV multispectral orthomosaics to assess crop damage caused by heavy winds, storms, and rains (lodging).",
        "tab_home": "🏠 Home",
        "tab_spectral": "📊 Spectral Analytics & Charts",
        "tab_calc": "🧮 Insurance Premium Calculator",
        "tab_resources": "🔗 Government Schemes & Resources",
        "tab_about": "ℹ️ About Us",
        "home_uploader_label": "Upload orthomosaic file(s)",
        "home_uploader_help": "Upload either a single 5-band GeoTIFF OR 5 separate single-band GeoTIFFs (Blue, Green, Red, RedEdge, NIR).",
        "home_loaded_single": "Loaded: {} (Expected 5-band TIFF)",
        "home_loaded_multi": "Loaded {} files: {}",
        "home_analyze_btn": "🔍 Analyze Field",
        "home_processing_spinner": "Processing orthomosaic — computing vegetation indices, running zonal statistics, and predicting...",
        "home_error_backend": "Error from backend: {}",
        "home_error_reach": "Could not reach backend: {}",
        "home_result_banner": "✅ Analysis complete — {} grid cells analyzed",
        "home_metric_total": "Total Area",
        "home_metric_lodged": "Lodged Area",
        "home_metric_non_lodged": "Non-Lodged Area",
        "home_metric_cells": "Lodged Cells",
        "home_map_title": "🗺️ Field Map",
        "home_map_legend": "🔴 Red = Lodged | 🟢 Green = Non-Lodged",
        "home_upload_prompt": "📤 Upload an orthomosaic and click <b>Analyze Field</b> to see results here.",
        "spectral_title": "📊 Spectral Analytics & Charts",
        "spectral_desc": "Select any band or computed Vegetation Index to visualize its value distribution and descriptive statistics across the field.",
        "spectral_select_label": "Select Band or Vegetation Index to analyze:",
        "spectral_range_info": "Visualizing values ranging from **{:.4f}** (Low) to **{:.4f}** (High).",
        "spectral_dist_title": "Value Distribution: {}",
        "spectral_stats_title": "Descriptive Statistics",
        "spectral_avg": "Average",
        "spectral_median": "Median",
        "spectral_min": "Min Value",
        "spectral_max": "Max Value",
        "spectral_warning": "⚠️ Please upload files and run the analysis in the **🏠 Home** tab first.",
        "calc_title": "🧮 Insurance Premium Calculator",
        "calc_subtitle": "Know your estimated crop insurance premium in just a few steps.",
        "calc_season": "Season *",
        "calc_year": "Year *",
        "calc_scheme": "Scheme *",
        "calc_state": "State *",
        "calc_district": "District *",
        "calc_district_placeholder": "Enter your District (e.g. Bhopal)",
        "calc_crop": "Crop *",
        "calc_area": "Area (Hectare) *",
        "calc_mandatory_note": "* All fields marked with * are mandatory. Please ensure all details are correct for accurate premium calculation.",
        "calc_btn_check": "Check Premium ➔",
        "calc_btn_reset": "🔄 Reset Form",
        "calc_error_fields": "Please fill in all mandatory fields (*) to calculate premium.",
        "calc_res_title": "📋 Premium Calculation Results",
        "calc_res_crop_season": "Crop / Season",
        "calc_res_location": "Location",
        "calc_res_area": "Insured Area",
        "calc_res_sum_ha": "Sum Insured (per Ha)",
        "calc_res_actuarial_rate": "Actuarial Premium Rate",
        "calc_res_farmer_rate": "Farmer Premium Rate",
        "calc_res_subsidy_rate": "Govt. Subsidy Rate",
        "calc_res_total_sum": "Total Sum Insured",
        "calc_res_farmer_payable": "Farmer Premium Payable",
        "calc_res_gov_subsidy": "Govt. Subsidy Share",
        "calc_crop_paddy": "Paddy (Rice)",
        "calc_crop_wheat": "Wheat",
        "calc_crop_maize": "Maize",
        "calc_crop_soyabean": "Soyabean",
        "calc_crop_cotton": "Cotton",
        "calc_crop_sugarcane": "Sugarcane",
        "calc_crop_mustard": "Mustard",
        "calc_crop_gram": "Gram (Chickpea)",
        "calc_crop_barley": "Barley",
        "res_title": "🔗 Government Schemes & Farmer Guidelines",
        "res_portals_header": "🌾 Key Government Portals",
        "res_portal_pmfby": "PM Fasal Bima Yojana (PMFBY) Portal — Official portal for crop insurance claims and premium checks.",
        "res_portal_pmkisan": "PM-KISAN Samman Nidhi Portal — Government scheme providing income support of ₹6,000/year to farmers.",
        "res_portal_enam": "e-NAM Portal — National online market portal for trading agricultural products.",
        "res_portal_soil": "Soil Health Card Scheme — Information on soil testing, health cards, and optimal fertilizer use.",
        "res_pdf_header": "📥 Official PDF Resources",
        "res_pdf_pmfby": "PMFBY Operational Guidelines PDF — Official government rulebook of PMFBY.",
        "res_pdf_wheat": "Wheat Lodging Prevention Guide (FAO) — Guidelines on preventing crop damage.",
        "res_pdf_icar": "ICAR Agricultural Advisory Services — ICAR portal for crop updates and storm advisories.",
        "res_mitigation_header": "💡 Prevention & Mitigation of Crop Lodging",
        "res_mitigation_desc": "Crop lodging occurs when plant stems bend or break, making harvesting difficult and reducing yields. Prevent it with these best practices:",
        "res_mitigation_1": "Select Lodging-Resistant Varieties: Choose crop varieties with shorter, stronger stems (semi-dwarf varieties).",
        "res_mitigation_2": "Balanced Fertilizer Application: Avoid excessive Nitrogen (N) application which leads to weak, elongated stems. Ensure adequate Potassium (K) to improve stem strength.",
        "res_mitigation_3": "Optimal Seed Rate: Do not sow seeds too densely. High plant density causes thin, fragile stalks.",
        "res_mitigation_4": "Smart Irrigation Timing: Avoid irrigating the field when high-speed winds are predicted, as wet soil reduces root anchoring.",
        "about_header": "ℹ️ About Us",
        "about_card_title": "About the Platform",
        "about_card_desc": "Crop lodging occurs when plant stems bend or break, making harvesting difficult and cutting down yields. This portal processes high-resolution UAV orthomosaics to automatically classify and quantify lodging. By calculating 26 vegetation indices across a 5m grid, it detects damage fields, supports PMFBY insurance claims, and offers crop management guidance.",
        "about_how_header": "⚙️ How it works",
        "about_how_step1": "1. Upload orthomosaic: Upload single or multi-band TIFF files of the field.",
        "about_how_step2": "2. Feature Extraction: The backend aligns bands, maps grid cells, and calculates 26 Vegetation Indices.",
        "about_how_step3": "3. Model Prediction: A trained Machine Learning model predicts lodging/non-lodging state.",
        "about_how_step4": "4. Visual Analytics: Charts and statistics display health analytics of the field.",
        "about_meta_header": "🔬 Model Metadata",
        "about_meta_arch": "Model Architecture",
        "about_meta_features": "Features Trained",
        "about_meta_status": "Status",
        "about_meta_region": "Deployment region",
        "about_meta_region_val": "Central India (Madhya Pradesh)",
        "about_footer": "Crop Lodging Detection Portal © 2026. Ministry of Agriculture & Farmers Welfare Alignment.",
        "auth_mode_label": "🔑 Portal Access Mode / मोड चुनें",
        "auth_mode_login": "Sign In (लॉगिन)",
        "auth_mode_register": "Sign Up (पंजीकरण)",
        "register_header": "📝 Register Farmer Account",
        "register_sub": "Enter details below to register your farmer profile.",
        "register_btn": "📝 Create Account",
        "register_success": "Registration successful! You can now Sign In.",
        "register_fail": "Username/Mobile already exists. Please choose a different one.",
        "login_fail": "Invalid Mobile Number or Password / गलत मोबाइल नंबर या पासवर्ड"
    },
    "Hindi": {
        "portal_title": "फसल नुकसान (लॉजिंग) डिटेक्शन पोर्टल",
        "portal_subtitle": "ड्रोन रिमोट सेंसिंग और विश्लेषण के साथ किसानों और बीमा कंपनियों का सशक्तिकरण",
        "lang_select": "Choose Language / भाषा चुनें",
        "login_header": "🔐 पोर्टल लॉगिन",
        "login_sub": "पोर्टल का उपयोग करने के लिए कृपया अपना मोबाइल नंबर और पासवर्ड दर्ज करें।",
        "username_label": "मोबाइल / उपयोगकर्ता नाम *",
        "username_placeholder": "मोबाइल नंबर दर्ज करें",
        "password_label": "पासवर्ड *",
        "password_placeholder": "पासवर्ड दर्ज करें",
        "login_btn": "🔑 लॉगिन करें",
        "login_err": "कृपया उपयोगकर्ता नाम और पासवर्ड दर्ज करें।",
        "logout_btn": "❌ लॉगआउट",
        "role_label": "पहुंच स्तर: किसान (भोपाल क्षेत्र)",
        "sidebar_info_title": "🌾 त्वरित जानकारी",
        "sidebar_info_desc": "यह प्लेटफॉर्म तेज हवाओं, आंधी और बारिश के कारण फसलों के गिरने (लॉजिंग) से होने वाले नुकसान का आकलन करने के लिए यूएवी मल्टीस्पेक्ट्रल ऑर्थोमोसैक का विश्लेषण करता है।",
        "tab_home": "🏠 मुख्य पृष्ठ",
        "tab_spectral": "📊 विश्लेषण और चार्ट",
        "tab_calc": "🧮 फसल बीमा प्रीमियम कैलकुलेटर",
        "tab_resources": "🔗 सरकारी योजनाएं और संसाधन",
        "tab_about": "ℹ️ हमारे बारे में",
        "home_uploader_label": "ऑर्थोमोसैक फाइल(फाइलें) अपलोड करें",
        "home_uploader_help": "या तो एक एकल 5-बैंड GeoTIFF अपलोड करें या 5 अलग-अलग सिंगल-बैंड GeoTIFF (ब्लू, ग्रीन, रेड, रेडएज, एनआईआर) अपलोड करें।",
        "home_loaded_single": "लोडेड: {} (अपेक्षित 5-बैंड TIFF)",
        "home_loaded_multi": "लोडेड {} फाइलें: {}",
        "home_analyze_btn": "🔍 खेत का विश्लेषण करें",
        "home_processing_spinner": "ऑर्थोमोसैक का प्रसंस्करण — वनस्पति सूचकांकों की गणना, क्षेत्रीय सांख्यिकी और भविष्यवाणी की जा रही है...",
        "home_error_backend": "बैकएंड से त्रुटि: {}",
        "home_error_reach": "बैकएंड तक नहीं पहुंचा जा सका: {}",
        "home_result_banner": "✅ विश्लेषण पूर्ण — {} ग्रिड कोशिकाओं का विश्लेषण किया गया",
        "home_metric_total": "कुल क्षेत्रफल",
        "home_metric_lodged": "प्रभावित (गिरा हुआ) क्षेत्र",
        "home_metric_non_lodged": "अप्रभावित क्षेत्र",
        "home_metric_cells": "प्रभावित ग्रिड कोशिकाएं",
        "home_map_title": "🗺️ खेत का नक्शा",
        "home_map_legend": "🔴 लाल = प्रभावित (गिरा हुआ) | 🟢 हरा = अप्रभावित",
        "home_upload_prompt": "📤 ऑर्थोमोसैक अपलोड करें और परिणाम देखने के लिए <b>खेत का विश्लेषण करें</b> पर क्लिक करें।",
        "spectral_title": "📊 स्पेक्ट्रल विश्लेषण और चार्ट",
        "spectral_desc": "खेत में इसके मूल्य वितरण और वर्णनात्मक सांख्यिकी को देखने के लिए किसी भी बैंड या वनस्पति सूचकांक का चयन करें।",
        "spectral_select_label": "विश्लेषण के लिए बैंड या वनस्पति सूचकांक चुनें:",
        "spectral_range_info": "मूल्य सीमा **{:.4f}** (कम) से **{:.4f}** (उच्च) का प्रदर्शन।",
        "spectral_dist_title": "मूल्य वितरण: {}",
        "spectral_stats_title": "वर्णनात्मक सांख्यिकी",
        "spectral_avg": "औसत (Average)",
        "spectral_median": "मध्यिका (Median)",
        "spectral_min": "न्यूनतम मूल्य",
        "spectral_max": "अधिकतम मूल्य",
        "spectral_warning": "⚠️ कृपया पहले **🏠 मुख्य पृष्ठ** टैब में फाइलें अपलोड करें और विश्लेषण चलाएं।",
        "calc_title": "🧮 फसल बीमा प्रीमियम कैलकुलेटर",
        "calc_subtitle": "बस कुछ ही चरणों में अपने अनुमानित फसल बीमा प्रीमियम की गणना करें।",
        "calc_season": "सत्र / मौसम *",
        "calc_year": "वर्ष *",
        "calc_scheme": "योजना *",
        "calc_state": "राज्य *",
        "calc_district": "जिला *",
        "calc_district_placeholder": "अपना जिला दर्ज करें (उदा. भोपाल)",
        "calc_crop": "फसल *",
        "calc_area": "क्षेत्रफल (हेक्टेयर) *",
        "calc_mandatory_note": "* सभी तारांकित (*) फ़ील्ड अनिवार्य हैं। सटीक प्रीमियम गणना के लिए कृपया सुनिश्चित करें कि सभी विवरण सही हैं।",
        "calc_btn_check": "प्रीमियम चेक करें ➔",
        "calc_btn_reset": "🔄 फॉर्म रीसेट करें",
        "calc_error_fields": "प्रीमियम की गणना करने के लिए कृपया सभी अनिवार्य (*) फ़ील्ड भरें।",
        "calc_res_title": "📋 प्रीमियम गणना परिणाम",
        "calc_res_crop_season": "फसल / मौसम",
        "calc_res_location": "स्थान",
        "calc_res_area": "बीमाकृत क्षेत्र",
        "calc_res_sum_ha": "बीमा राशि (प्रति हेक्टेयर)",
        "calc_res_actuarial_rate": "एक्चुअरियल प्रीमियम दर",
        "calc_res_farmer_rate": "किसान प्रीमियम दर",
        "calc_res_subsidy_rate": "सरकारी सब्सिडी दर",
        "calc_res_total_sum": "कुल बीमा राशि",
        "calc_res_farmer_payable": "किसान द्वारा देय प्रीमियम",
        "calc_res_gov_subsidy": "सरकारी सब्सिडी का हिस्सा",
        "calc_crop_paddy": "धान (चावल)",
        "calc_crop_wheat": "गेहूं",
        "calc_crop_maize": "मक्का",
        "calc_crop_soyabean": "सोयाबीन",
        "calc_crop_cotton": "कपास",
        "calc_crop_sugarcane": "गन्ना",
        "calc_crop_mustard": "सरसों",
        "calc_crop_gram": "चना",
        "calc_crop_barley": "जौ",
        "res_title": "🔗 सरकारी योजनाएं और किसान दिशा-निर्देश",
        "res_portals_header": "🌾 प्रमुख सरकारी पोर्टल",
        "res_portal_pmfby": "प्रधानमंत्री फसल बीमा योजना (PMFBY) पोर्टल — फसल बीमा दावों और प्रीमियम चेक के लिए आधिकारिक पोर्टल।",
        "res_portal_pmkisan": "पीएम-किसान सम्मान निधि पोर्टल — किसानों को ₹6,000/वर्ष की आय सहायता प्रदान करने वाली सरकारी योजना।",
        "res_portal_enam": "ई-नाम पोर्टल — कृषि उत्पादों के व्यापार के लिए राष्ट्रीय ऑनलाइन बाजार पोर्टल।",
        "res_portal_soil": "मृदा स्वास्थ्य कार्ड योजना — मिट्टी परीक्षण, स्वास्थ्य कार्ड और उर्वरक उपयोग पर जानकारी।",
        "res_pdf_header": "📥 आधिकारिक पीडीएफ संसाधन",
        "res_pdf_pmfby": "PMFBY परिचालन दिशानिर्देश पीडीएफ — PMFBY की आधिकारिक सरकारी नियमपुस्तिका।",
        "res_pdf_wheat": "गेहूं लॉजिंग रोकथाम गाइड (FAO) — फसल क्षति को रोकने पर दिशानिर्देश।",
        "res_pdf_icar": "ICAR कृषि सलाहकार सेवाएं — फसल अपडेट और आंधी-तूफान सलाह के लिए ICAR पोर्टल।",
        "res_mitigation_header": "💡 फसल गिरने (लॉजिंग) से बचाव और रोकथाम",
        "res_mitigation_desc": "फसल तब गिरती है जब पौधे के तने झुक जाते हैं या टूट जाते हैं, जिससे कटाई कठिन हो जाती है और पैदावार कम हो जाती है। इन सर्वोत्तम प्रथाओं से इसे रोकें:",
        "res_mitigation_1": "गिरने-प्रतिरोधी किस्मों का चयन करें: छोटे और मजबूत तनों वाली फसल किस्मों का चयन करें (अर्ध-बौनी किस्में)।",
        "res_mitigation_2": "संतुलित उर्वरक अनुप्रयोग: अत्यधिक नाइट्रोजन (N) के उपयोग से बचें जिससे तने कमजोर और लंबे हो जाते हैं। तने की मजबूती के लिए पर्याप्त पोटेशियम (K) सुनिश्चित करें।",
        "res_mitigation_3": "इष्टतम बीज दर: बहुत घनी बुआई न करें। उच्च पौधे घनत्व से पतले और नाजुक तने बनते हैं।",
        "res_mitigation_4": "बुद्धिमानी से सिंचाई का समय: जब तेज हवाओं की भविष्यवाणी की गई हो, तो खेत की सिंचाई करने से बचें, क्योंकि गीली मिट्टी जड़ों की पकड़ को कमजोर करती।",
        "about_header": "ℹ️ हमारे बारे में",
        "about_card_title": "प्लेटफॉर्म के बारे में",
        "about_card_desc": "फसल तब गिरती है जब पौधे के तने झुक जाते हैं या टूट जाते हैं, जिससे कटाई कठिन हो जाती है और पैदावार कम हो जाती है। यह पोर्टल ऑर्थोमोसैक का विश्लेषण करके फसल गिरने का वर्गीकरण करता है। यह 5 मीटर के ग्रिड पर 26 वनस्पति सूचकांकों की गणना करके क्षतिग्रस्त क्षेत्रों का पता लगाता है, PMFBY बीमा दावों में सहायता करता है, और फसल प्रबंधन के सुझाव देता है।",
        "about_how_header": "⚙️ यह कैसे काम करता है",
        "about_how_step1": "1. ऑर्थोमोसैक अपलोड करें: खेत की सिंगल या मल्टी-बैंड TIFF फाइलें अपलोड करें।",
        "about_how_step2": "2. विशेषताएं निकालना: बैकएंड बैंड को संरेखित करता है, ग्रिड मैप करता है और 26 वनस्पति सूचकांकों की गणना करता।",
        "about_how_step3": "3. मॉडल भविष्यवाणी: एक प्रशिक्षित मशीन लर्निंग मॉडल प्रभावित/गैर-प्रभावित स्थिति की भविष्यवाणी करता है।",
        "about_how_step4": "4. दृश्य विश्लेषण: चार्ट और सांख्यिकी खेत के स्वास्थ्य विश्लेषण को प्रदर्शित करते हैं।",
        "about_meta_header": "🔬 मॉडल मेटाडेटा",
        "about_meta_arch": "मॉडल आर्किटेक्चर",
        "about_meta_features": "प्रशिक्षित विशेषताएं",
        "about_meta_status": "स्थिति",
        "about_meta_region": "तैनाती क्षेत्र",
        "about_meta_region_val": "मध्य भारत (मध्य प्रदेश)",
        "about_footer": "फसल नुकसान (लॉजिंग) पोर्टल © 2026. कृषि और किसान कल्याण मंत्रालय।",
        "auth_mode_label": "🔑 प्रवेश विकल्प / मोड चुनें",
        "auth_mode_login": "लॉगिन (प्रवेश)",
        "auth_mode_register": "नया पंजीकरण",
        "register_header": "📝 किसान नया खाता बनाएं",
        "register_sub": "अपना किसान प्रोफ़ाइल पंजीकृत करने के लिए नीचे विवरण दर्ज करें।",
        "register_btn": "📝 खाता बनाएं",
        "register_success": "पंजीकरण सफल! अब आप लॉगिन कर सकते हैं।",
        "register_fail": "मोबाइल नंबर/उपयोगकर्ता नाम पहले से मौजूद है। कृपया दूसरा चुनें।",
        "login_fail": "गलत मोबाइल नंबर या पासवर्ड"
    }
}

# --- Initialize Session States ---
# Load login credentials from URL query parameters if present (persists login on refresh)
if "username" in st.query_params:
    st.session_state.logged_in = True
    st.session_state.username = st.query_params["username"]

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""
if "language" not in st.session_state:
    st.session_state.language = "English"
if "results" not in st.session_state:
    st.session_state.results = None
if "calc_results" not in st.session_state:
    st.session_state.calc_results = None

# --- Custom CSS (With anchor link hides, instruction hides, and larger text) ---
st.markdown("""
<style>
    .main {
        background-color: transparent !important;
    }
    .stApp {
        background-image: linear-gradient(rgba(245, 249, 244, 0.84), rgba(235, 244, 230, 0.84)), url('https://images.unsplash.com/photo-1625246333195-78d9c38ad449?q=80&w=1920');
        background-size: cover;
        background-attachment: fixed;
        background-position: center;
    }
    
    /* Hide default Streamlit header anchor links and actions on hover */
    h1 a, h2 a, h3 a, h4 a, h5 a, h6 a,
    .stApp a[href^="#"], 
    .stApp a.header-anchor, 
    .stApp [data-testid="stHeaderActionElements"] {
        display: none !important;
        visibility: hidden !important;
        opacity: 0 !important;
        width: 0 !important;
        height: 0 !important;
    }
    
    /* Hide the "Press Enter to submit form" caption from input fields inside forms */
    div[data-testid="InputInstructions"] {
        display: none !important;
    }
    
    /* Global text colors for high visibility */
    h1, h2, h3, h4, h5, h6 {
        color: #2e5339 !important;
        font-weight: 700 !important;
    }
    p, li, span, label {
        color: #2e5339 !important;
        font-size: 1.02rem !important;
        font-weight: 500;
    }
    
    /* Input fields and labels override to ensure readability */
    .stSelectbox label, .stNumberInput label, .stTextInput label, div[data-testid="stMarkdownContainer"] p {
        color: #2e5339 !important;
        font-weight: 600 !important;
    }
    
    /* Interactive Government Portal Hero Banner */
    .hero-banner {
        background-image: linear-gradient(rgba(0, 0, 0, 0.45), rgba(0, 0, 0, 0.45)), url('https://images.unsplash.com/photo-1500937386664-56d1dfef3854?q=80&w=1400');
        background-size: cover;
        background-position: center;
        padding: 55px 35px;
        border-radius: 16px;
        text-align: center;
        margin-bottom: 25px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.12);
        border: 2px solid rgba(255, 255, 255, 0.3);
    }
    
    /* Force Hero Banner Title and P to be WHITE at all times and not get overridden by global rules */
    .stApp .hero-title,
    .hero-title {
        font-size: 3.8rem !important; /* Large, bold title */
        font-weight: 900 !important; /* Extrabold */
        color: #ffffff !important; /* White color guaranteed */
        margin: 0 !important;
        text-shadow: 2px 2px 7px rgba(0, 0, 0, 0.85) !important;
        line-height: 1.2 !important;
    }
    .stApp .hero-banner p,
    div.hero-banner p {
        font-size: 1.35rem !important; /* Larger subtitle */
        color: #f5fcf7 !important; /* White color guaranteed */
        margin-top: 12px !important;
        margin-bottom: 0 !important;
        text-shadow: 1px 1px 4px rgba(0, 0, 0, 0.6) !important;
        font-weight: 600 !important;
    }
    
    /* Centered Card for About section */
    .about-card {
        max-width: 950px;
        margin: 10px auto 25px auto;
        padding: 30px;
        background-color: rgba(255, 255, 255, 0.96);
        border-radius: 16px;
        box-shadow: 0 6px 18px rgba(0,0,0,0.08);
        border: 1px solid #d0dec3;
    }
    
    /* Style Streamlit native bordered containers as white cards (Login Card) */
    div[data-testid="stVerticalBlockBorder"],
    div[class*="stVerticalBlockBorder"],
    div[data-testid="stVerticalBlock"] > div[style*="border"],
    div[data-testid="stVerticalBlock"] > div[style*="Border"] {
        background-color: rgba(255, 255, 255, 0.96) !important;
        border-radius: 16px !important;
        border: 1.5px solid #d0dec3 !important;
        padding: 30px !important;
        box-shadow: 0 6px 18px rgba(0,0,0,0.08) !important;
        margin-top: 50px !important;
    }
    
    /* Center the Streamlit tab bar headers and items */
    div[data-testid="stTabBar"], 
    div[data-testid="stTabBar"] > div,
    div[data-testid="stTabBar"] > div > div,
    div[role="tablist"] {
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
        margin: 0 auto !important;
    }
    
    /* Hide Leaflet random technical attribution details at bottom of maps */
    .leaflet-control-attribution {
        display: none !important;
    }
    
    /* Streamlit Tab Styles for High Visibility */
    button[data-baseweb="tab"] {
        background-color: transparent !important;
        padding: 10px 20px !important;
        border-bottom-color: transparent !important;
        white-space: nowrap !important;
    }
    button[data-baseweb="tab"] p, button[data-baseweb="tab"] span, button[data-baseweb="tab"] div {
        color: #5a6b5d !important;
        font-size: 1.12rem !important;
        font-weight: 600 !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        border-bottom: 3.5px solid #2e5339 !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] p, 
    button[data-baseweb="tab"][aria-selected="true"] span, 
    button[data-baseweb="tab"][aria-selected="true"] div {
        color: #2e5339 !important;
        font-weight: 800 !important;
    }
    button[data-baseweb="tab"]:hover p, 
    button[data-baseweb="tab"]:hover span, 
    button[data-baseweb="tab"]:hover div {
        color: #2e5339 !important;
    }
    
    /* Warning / Info Box alert overrides to ensure dark brown readability */
    div[data-testid="stNotification"] *, div[class*="stAlert"] * {
        color: #856404 !important;
    }
    
    /* Metrics panel values styling */
    div[data-testid="stMetric"] {
        background-color: rgba(255, 255, 255, 0.95);
        border: 1px solid #e0e6db;
        border-radius: 14px;
        padding: 18px 16px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }
    div[data-testid="stMetric"] label, div[data-testid="stMetric"] label * {
        color: #5a6b5d !important;
        font-weight: 600;
    }
    div[data-testid="stMetricValue"], div[data-testid="stMetricValue"] * {
        color: #2e5339 !important;
        font-weight: 800 !important;
        font-size: 1.8rem !important;
    }
    div[data-testid="stMetricDelta"], div[data-testid="stMetricDelta"] * {
        color: #4a7856 !important;
        font-weight: bold !important;
    }
    
    .upload-card {
        background-color: rgba(255, 255, 255, 0.95);
        border: 1.5px dashed #9db98c;
        border-radius: 16px;
        padding: 24px;
        text-align: center;
    }
    .stApp .upload-card,
    .stApp .upload-card p,
    .stApp .upload-card span,
    .stApp .upload-card b,
    .stApp .upload-card div {
        color: #2e5339 !important;
        font-weight: 600 !important;
        font-size: 1.15rem !important;
    }
    
    /* Results banner text overrides */
    .result-banner {
        background: linear-gradient(90deg, #2e5339 0%, #4a7856 100%);
        padding: 16px 24px;
        border-radius: 14px;
        margin-bottom: 20px;
        font-size: 1.1rem;
        font-weight: 600;
    }
    .result-banner, .result-banner * {
        color: white !important;
    }
    
    /* Action and Form Submit buttons styling */
    .stButton>button, .stFormSubmitButton>button {
        background-color: #2e5339 !important;
        border-radius: 10px !important;
        padding: 10px 28px !important;
        font-weight: 600 !important;
        border: none !important;
    }
    .stButton>button:hover, .stFormSubmitButton>button:hover {
        background-color: #3f6b4d !important;
    }
    
    /* Absolute override to ensure button text is ALWAYS white and not overridden by stApp p/span */
    .stApp button, 
    .stApp button *, 
    .stApp button p, 
    .stApp button span, 
    .stApp button div,
    .stApp .stButton button p,
    .stApp .stFormSubmitButton button p {
        color: #ffffff !important;
    }
    
    /* Style Streamlit forms to look like PMFBY portal cards */
    div[data-testid="stForm"] {
        background-color: rgba(255, 255, 255, 0.95) !important;
        border-radius: 14px !important;
        border: 1px solid #d4decb !important;
        padding: 25px !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05) !important;
        margin-bottom: 20px !important;
    }
    
    section[data-testid="stSidebar"] {
        background-color: #2e5339;
    }
    section[data-testid="stSidebar"] *,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] li,
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] h4,
    section[data-testid="stSidebar"] h5,
    section[data-testid="stSidebar"] h6 {
        color: #f0f4ec !important;
    }
</style>
""", unsafe_allow_html=True)

# --- 1. GATED AUTHENTICATION SCREEN (Sign In / Sign Up) ---
if not st.session_state.logged_in:
    _, col_login, _ = st.columns([1, 1.6, 1])
    
    with col_login:
        # Wrap the login card elements inside a native Streamlit container with a border
        with st.container(border=True):
            # Language Selectbox inside Card
            selected_lang = st.selectbox(
                "Select Language / भाषा चुनें *",
                ["English", "Hindi"],
                index=0 if st.session_state.language == "English" else 1,
                key="login_language_select"
            )
            st.session_state.language = selected_lang
            t = TRANSLATIONS[st.session_state.language]
            
            # Toggle Sign In / Sign Up Mode
            auth_mode = st.radio(
                t["auth_mode_label"],
                [t["auth_mode_login"], t["auth_mode_register"]],
                horizontal=True,
                key="auth_mode_radio"
            )
            
            if auth_mode == t["auth_mode_login"]:
                st.markdown(f"### {t['login_header']}")
                st.markdown(f"<p style='color: #4a5d4e;'>{t['login_sub']}</p>", unsafe_allow_html=True)
                
                with st.form(key="front_login_form", border=False):
                    username = st.text_input(t["username_label"], placeholder=t["username_placeholder"], key="gated_user")
                    password = st.text_input(t["password_label"], type="password", placeholder=t["password_placeholder"], key="gated_pass")
                    login_submitted = st.form_submit_button(t["login_btn"])
                    
                    if login_submitted:
                        if username and password:
                            username_clean = username.strip()
                            password_clean = password.strip()
                            if validate_user(username_clean, password_clean):
                                st.session_state.logged_in = True
                                st.session_state.username = username_clean
                                st.query_params["username"] = username_clean
                                st.rerun()
                            else:
                                st.error(t["login_fail"])
                        else:
                            st.error(t["login_err"])
            else:
                st.markdown(f"### {t['register_header']}")
                st.markdown(f"<p style='color: #4a5d4e;'>{t['register_sub']}</p>", unsafe_allow_html=True)
                
                with st.form(key="front_register_form", border=False):
                    reg_username = st.text_input(t["username_label"], placeholder=t["username_placeholder"], key="reg_user")
                    reg_password = st.text_input(t["password_label"], type="password", placeholder=t["password_placeholder"], key="reg_pass")
                    register_submitted = st.form_submit_button(t["register_btn"])
                    
                    if register_submitted:
                        if reg_username and reg_password:
                            reg_user_clean = reg_username.strip()
                            reg_pass_clean = reg_password.strip()
                            if register_user(reg_user_clean, reg_pass_clean):
                                st.success(t["register_success"])
                            else:
                                st.error(t["register_fail"])
                        else:
                            st.error(t["login_err"])
                            
    st.stop()  # Stop execution so non-authenticated users see only login

# --- Extract Current Language Translations ---
t = TRANSLATIONS[st.session_state.language]

# --- 2. SIDEBAR (Only visible when logged in) ---
with st.sidebar:
    st.success(f"✔️ Logged in as: **{st.session_state.username}**")
    
    # Language Switcher in Sidebar
    st.markdown("---")
    sidebar_lang = st.selectbox(
        "🌐 Language / भाषा",
        ["English", "Hindi"],
        index=0 if st.session_state.language == "English" else 1,
        key="sidebar_language_select"
    )
    if sidebar_lang != st.session_state.language:
        st.session_state.language = sidebar_lang
        st.rerun()
        
    if st.button(t["logout_btn"], key="logout_btn_sidebar"):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.session_state.results = None
        st.session_state.calc_results = None
        st.query_params.clear()
        st.rerun()

    st.markdown("---")
    st.markdown(f"### {t['sidebar_info_title']}")
    st.markdown(t["sidebar_info_desc"])

# --- Center Hero Banner ---
st.markdown(f"""
<div class="hero-banner">
    <p class="hero-title">{t['portal_title']}</p>
    <p>{t['portal_subtitle']}</p>
</div>
""", unsafe_allow_html=True)

# --- Define Dashboard Tabs ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    t["tab_home"],
    t["tab_spectral"],
    t["tab_calc"],
    t["tab_resources"],
    t["tab_about"]
])

with tab1:
    # --- File upload ---
    uploaded_files = st.file_uploader(
        t["home_uploader_label"],
        type=["tif", "tiff"],
        accept_multiple_files=True,
        help=t["home_uploader_help"],
        key="uploader_lodging"
    )

    if uploaded_files:
        if len(uploaded_files) == 1:
            st.info(t["home_loaded_single"].format(uploaded_files[0].name))
        else:
            st.info(t["home_loaded_multi"].format(len(uploaded_files), ", ".join([f"`{f.name}`" for f in uploaded_files])))

    col_a, col_b = st.columns([1, 4])
    with col_a:
        analyze_clicked = st.button(t["home_analyze_btn"], disabled=(not uploaded_files))

    if analyze_clicked and uploaded_files:
        with st.spinner(t["home_processing_spinner"]):
            try:
                files = [("files", (f.name, f.getvalue(), "image/tiff")) for f in uploaded_files]
                response = requests.post(f"{BACKEND_URL}/predict", files=files, timeout=300)
                if response.status_code == 200:
                    res_data = response.json()
                    if "geojson" in res_data:
                        res_data["geojson"] = fix_geojson_coordinates(res_data["geojson"])
                    st.session_state.results = res_data
                else:
                    error_detail = response.json().get('error', response.text)
                    st.error(t["home_error_backend"].format(error_detail))
                    st.session_state.results = None
            except Exception as e:
                st.error(t["home_error_reach"].format(e))
                st.session_state.results = None

    # --- Results ---
    if st.session_state.results:
        summary = st.session_state.results["summary"]
        geojson = fix_geojson_coordinates(st.session_state.results["geojson"])

        st.markdown(
            f'<div class="result-banner">{t["home_result_banner"].format(summary["total_cells"])}</div>',
            unsafe_allow_html=True,
        )

        m1, m2, m3, m4 = st.columns(4)
        m1.metric(t["home_metric_total"], f"{summary['total_area_sqm']:.0f} m²")
        m2.metric(t["home_metric_lodged"], f"{summary['lodged_area_sqm']:.0f} m²", f"{summary['lodged_percentage']}%")
        m3.metric(t["home_metric_non_lodged"], f"{summary['non_lodged_area_sqm']:.0f} m²", f"{100 - summary['lodged_percentage']:.1f}%")
        m4.metric(t["home_metric_cells"], f"{summary['lodged_cells']} / {summary['total_cells']}")

        st.markdown(f"### {t['home_map_title']}")
        st.markdown(f"<p style='color: #2e5339; font-weight: bold; font-size: 1.15rem; margin-bottom: 15px;'>{t['home_map_legend']}</p>", unsafe_allow_html=True)

        # Center the map and fit bounds to the geojson
        lons = [c[0] for f in geojson["features"] for c in f["geometry"]["coordinates"][0]]
        lats = [c[1] for f in geojson["features"] for c in f["geometry"]["coordinates"][0]]
        center = [sum(lats) / len(lats), sum(lons) / len(lons)]

        # Pre-calculate lodging colors in Python to avoid JS translation errors
        for f in geojson["features"]:
            pred = f["properties"].get("prediction", "non-lodged")
            f["properties"]["lodging_color"] = "#d9534f" if pred == "lodged" else "#4a9d5b"

        fmap = folium.Map(location=center, zoom_start=18, tiles=None)
        folium.TileLayer(
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attr=" ",  # Remove Esri attribution text
            name="Esri World Imagery"
        ).add_to(fmap)
        fmap.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]])

        folium.GeoJson(
            geojson,
            style_function=lambda feature: {
                "fillColor": feature["properties"]["lodging_color"],
                "color": "#ffffff",  # Distinct white borders
                "weight": 1.5,
                "fillOpacity": 0.55
            },
            tooltip=folium.GeoJsonTooltip(fields=["prediction", "confidence"],
                                           aliases=["Status:", "Confidence:"]),
        ).add_to(fmap)

        st_folium(fmap, width=None, height=550, key=f"folium_map_{summary['total_cells']}_{summary['lodged_cells']}")

    else:
        st.markdown(
            f'<div class="upload-card">{t["home_upload_prompt"]}</div>',
            unsafe_allow_html=True,
        )

with tab2:
    if st.session_state.results:
        geojson = fix_geojson_coordinates(st.session_state.results["geojson"])
        summary = st.session_state.results["summary"]
        
        st.markdown(f"### {t['spectral_title']}")
        st.markdown(t["spectral_desc"])
        
        first_props = geojson["features"][0]["properties"]
        features_available = sorted([k for k in first_props.keys() if k not in ["prediction", "confidence", "lodging_color", "feature_color", "feature_opacity"]])
        
        default_idx = features_available.index("NDVI") if "NDVI" in features_available else 0
        selected_feature = st.selectbox(
            t["spectral_select_label"],
            features_available,
            index=default_idx,
            key="analytics_feature_select"
        )
        
        vals = [f["properties"].get(selected_feature) for f in geojson["features"] if f["properties"].get(selected_feature) is not None]
        
        if vals:
            # Map removed as requested — rendering full-width Chart and Descriptive statistics
            st.markdown(f"#### {t['spectral_dist_title'].format(selected_feature)}")
            min_val = min(vals)
            max_val = max(vals)
            st.info(t["spectral_range_info"].format(min_val, max_val))
            
            counts, bins = np.histogram(vals, bins=15)
            bin_centers = 0.5 * (bins[:-1] + bins[1:])
            hist_df = pd.DataFrame({"Cell Count": counts}, index=[f"{b:.3f}" for b in bin_centers])
            
            st.bar_chart(hist_df, use_container_width=True)
            
            st.markdown(f"##### {t['spectral_stats_title']}")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric(t["spectral_avg"], f"{np.mean(vals):.4f}")
            c2.metric(t["spectral_median"], f"{np.median(vals):.4f}")
            c3.metric(t["spectral_min"], f"{np.min(vals):.4f}")
            c4.metric(t["spectral_max"], f"{np.max(vals):.4f}")
        else:
            st.error("No valid data found for the selected feature.")
    else:
        st.warning(t["spectral_warning"])

with tab3:
    st.markdown(f"### {t['calc_title']}")
    st.markdown(t["calc_subtitle"])
    
    # State-to-District mapping dictionary
    STATE_DISTRICTS = {
        "Select": ["Select State First"],
        "Madhya Pradesh": ["Bhopal", "Indore", "Gwalior", "Jabalpur", "Ujjain", "Sagar", "Rewa", "Vidisha", "Sehore", "Hoshangabad"],
        "Uttar Pradesh": ["Lucknow", "Kanpur", "Varanasi", "Agra", "Meerut", "Prayagraj", "Bareilly", "Aligarh", "Gorakhpur", "Jhansi"],
        "Maharashtra": ["Mumbai", "Pune", "Nagpur", "Thane", "Nashik", "Aurangabad", "Solapur", "Amravati", "Kolhapur", "Sangli"],
        "Rajasthan": ["Jaipur", "Jodhpur", "Udaipur", "Kota", "Bikaner", "Ajmer", "Alwar", "Sikar", "Bhilwara", "Jhunjhunu"],
        "Gujarat": ["Ahmedabad", "Surat", "Vadodara", "Rajkot", "Bhavnagar", "Jamnagar", "Gandhinagar", "Junagadh", "Anand", "Mehsana"]
    }

    # Premium Form Layout using a native Streamlit border container (allows dynamic dropdown reloading)
    with st.container(border=True):
        c_r1_1, c_r1_2, c_r1_3 = st.columns(3)
        with c_r1_1:
            season = st.selectbox(t["calc_season"], ["Select", "Kharif", "Rabi"], index=0)
        with c_r1_2:
            year = st.selectbox(t["calc_year"], ["Select", "2026", "2025", "2024", "2023"], index=0)
        with c_r1_3:
            scheme = st.selectbox(t["calc_scheme"], ["Select", "PMFBY (Pradhan Mantri Fasal Bima Yojana)", "WBCIS (Weather Based Crop Insurance Scheme)"], index=0)
            
        c_r2_1, c_r2_2, c_r2_3 = st.columns(3)
        with c_r2_1:
            state = st.selectbox(t["calc_state"], ["Select", "Madhya Pradesh", "Uttar Pradesh", "Maharashtra", "Rajasthan", "Gujarat"], index=0)
        with c_r2_2:
            # Dynamic dependent dropdown list of districts
            dist_options = STATE_DISTRICTS.get(state, ["Select State First"])
            district = st.selectbox(t["calc_district"], ["Select"] + dist_options if state != "Select" else dist_options, index=0)
            
        with c_r2_3:
            # Crop List directly populated with actual crop names (localized)
            crop_options = [
                "Select",
                t["calc_crop_paddy"],
                t["calc_crop_wheat"],
                t["calc_crop_maize"],
                t["calc_crop_soyabean"],
                t["calc_crop_cotton"],
                t["calc_crop_sugarcane"],
                t["calc_crop_mustard"],
                t["calc_crop_gram"],
                t["calc_crop_barley"]
            ]
            crop_selected = st.selectbox(t["calc_crop"], crop_options, index=0)
            
        c_r3_1, c_r3_2, c_r3_3 = st.columns(3)
        with c_r3_1:
            area_ha = st.number_input(t["calc_area"], min_value=0.1, max_value=100.0, value=2.0, step=0.5)
        with c_r3_2:
            st.write("") # Spacer
        with c_r3_3:
            st.write("") # Spacer
            
        st.write("---")
        st.write(f'<p style="color: #856404; font-size: 0.95rem; font-weight:600;">{t["calc_mandatory_note"]}</p>', unsafe_allow_html=True)
        
        c_btn_1, _ = st.columns([1, 4])
        with c_btn_1:
            check_premium_clicked = st.button(t["calc_btn_check"])
            
    # Reset Button Outside the Container
    if st.button(t["calc_btn_reset"]):
        st.session_state.calc_results = None
        st.rerun()

    # Premium Calculations
    if check_premium_clicked:
        if season == "Select" or year == "Select" or scheme == "Select" or state == "Select" or district == "Select" or district == "Select State First" or crop_selected == "Select":
            st.error(t["calc_error_fields"])
            st.session_state.calc_results = None
        else:
            # Map selected localized crop back to standard rules
            crop_rules = {
                t["calc_crop_paddy"]: {"sum_insured": 65000, "farmer_rate": 2.0, "total_rate": 12.0, "eng_name": "Paddy (Rice)"},
                t["calc_crop_maize"]: {"sum_insured": 45000, "farmer_rate": 2.0, "total_rate": 10.0, "eng_name": "Maize"},
                t["calc_crop_soyabean"]: {"sum_insured": 50000, "farmer_rate": 2.0, "total_rate": 11.0, "eng_name": "Soyabean"},
                t["calc_crop_cotton"]: {"sum_insured": 80000, "farmer_rate": 5.0, "total_rate": 15.0, "eng_name": "Cotton"},
                t["calc_crop_sugarcane"]: {"sum_insured": 90000, "farmer_rate": 5.0, "total_rate": 14.0, "eng_name": "Sugarcane"},
                t["calc_crop_wheat"]: {"sum_insured": 55000, "farmer_rate": 1.5, "total_rate": 9.5, "eng_name": "Wheat"},
                t["calc_crop_mustard"]: {"sum_insured": 48000, "farmer_rate": 1.5, "total_rate": 10.0, "eng_name": "Mustard"},
                t["calc_crop_gram"]: {"sum_insured": 42000, "farmer_rate": 1.5, "total_rate": 9.0, "eng_name": "Gram (Chickpea)"},
                t["calc_crop_barley"]: {"sum_insured": 40000, "farmer_rate": 1.5, "total_rate": 8.5, "eng_name": "Barley"}
            }
            
            rule = crop_rules.get(crop_selected, {"sum_insured": 50000, "farmer_rate": 2.0, "total_rate": 12.0, "eng_name": crop_selected})
            
            total_sum_insured = rule["sum_insured"] * area_ha
            actuarial_premium = total_sum_insured * (rule["total_rate"] / 100.0)
            farmer_premium = total_sum_insured * (rule["farmer_rate"] / 100.0)
            gov_subsidy = actuarial_premium - farmer_premium
            
            st.session_state.calc_results = {
                "crop": crop_selected,
                "season": season,
                "year": year,
                "state": state,
                "district": district,
                "area": area_ha,
                "sum_insured_ha": rule["sum_insured"],
                "total_sum_insured": total_sum_insured,
                "farmer_rate": rule["farmer_rate"],
                "total_rate": rule["total_rate"],
                "farmer_premium": farmer_premium,
                "gov_subsidy": gov_subsidy,
                "actuarial_premium": actuarial_premium
            }

    # Show Calculation output in a neat styled layout
    if st.session_state.calc_results:
        res = st.session_state.calc_results
        st.markdown(f"### {t['calc_res_title']}")
        
        c_res1, c_res2 = st.columns(2)
        with c_res1:
            st.markdown(f"""
            * **{t['calc_res_crop_season']}:** {res['crop']} ({res['season']} - {res['year']})
            * **{t['calc_res_location']}:** {res['district']}, {res['state']}
            * **{t['calc_res_area']}:** {res['area']:.2f} Hectares
            * **{t['calc_res_sum_ha']}:** ₹ {res['sum_insured_ha']:,.2f}
            """)
        with c_res2:
            st.markdown(f"""
            * **{t['calc_res_actuarial_rate']}:** {res['total_rate']}%
            * **{t['calc_res_farmer_rate']}:** {res['farmer_rate']}%
            * **{t['calc_res_subsidy_rate']}:** {res['total_rate'] - res['farmer_rate']:.1f}%
            """)
            
        st.write("---")
        
        m_r1, m_r2, m_r3 = st.columns(3)
        m_r1.metric(t["calc_res_total_sum"], f"₹ {res['total_sum_insured']:,.2f}")
        m_r2.metric(t["calc_res_farmer_payable"], f"₹ {res['farmer_premium']:,.2f}", f"Rate: {res['farmer_rate']}%")
        m_r3.metric(t["calc_res_gov_subsidy"], f"₹ {res['gov_subsidy']:,.2f}", f"Rate: {res['total_rate'] - res['farmer_rate']:.1f}%")

with tab4:
    st.markdown(f"### {t['res_title']}")
    
    col_schemes, col_pdf = st.columns([1, 1])
    
    with col_schemes:
        st.markdown(f"#### {t['res_portals_header']}")
        st.markdown(f"""
        - **[PM Fasal Bima Yojana (PMFBY) Portal](https://pmfby.gov.in/)** — {t['res_portal_pmfby']}
        - **[PM-KISAN Samman Nidhi Portal](https://pmkisan.gov.in/)** — {t['res_portal_pmkisan']}
        - **[e-NAM Portal](https://www.enam.gov.in/)** — {t['res_portal_enam']}
        - **[Soil Health Card Scheme](https://soilhealth.dac.gov.in/)** — {t['res_portal_soil']}
        """)
        
        st.markdown(f"#### {t['res_pdf_header']}")
        st.markdown(f"""
        - **[PMFBY Operational Guidelines PDF](https://pmfby.gov.in/pdf/Revised_Operational_Guidelines.pdf)** — {t['res_pdf_pmfby']}
        - **[Wheat Lodging Prevention Guide (FAO)](https://www.fao.org/3/x5872e/x5872e00.htm)** — {t['res_pdf_wheat']}
        - **[ICAR Agricultural Advisory Services](https://icar.org.in/)** — {t['res_pdf_icar']}
        """)
        
    with col_pdf:
        st.markdown(f"#### {t['res_mitigation_header']}")
        st.markdown(f"""
        {t['res_mitigation_desc']}
        
        1. **{t['res_mitigation_1']}**
        2. **{t['res_mitigation_2']}**
        3. **{t['res_mitigation_3']}**
        4. **{t['res_mitigation_4']}**
        """)

with tab5:
    st.markdown(f"## {t['about_header']}")
    
    # About details card centered
    st.markdown(f"""
    <div class="about-card">
        <h3 style="margin-top:0; color:#2e5339;">{t['about_card_title']}</h3>
        <p style="margin-bottom:0;">
            {t['about_card_desc']}
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Model specifications and How it works section
    col_about1, col_about2 = st.columns([1, 1])
    
    with col_about1:
        st.markdown(f"### {t['about_how_header']}")
        st.markdown(f"""
        * **{t['about_how_step1']}**
        * **{t['about_how_step2']}**
        * **{t['about_how_step3']}**
        * **{t['about_how_step4']}**
        """)
        
    with col_about2:
        st.markdown(f"### {t['about_meta_header']}")
        try:
            info = requests.get(f"{BACKEND_URL}/").json()
            st.markdown(f"""
            * **{t['about_meta_arch']}:** `{info.get('model', 'SVM_Linear')}`
            * **{t['about_meta_features']}:** `{info.get('num_features', '31')}` features (bands + VIs)
            * **{t['about_meta_status']}:** Operational ✅
            * **{t['about_meta_region']}:** {t['about_meta_region_val']}
            """)
        except Exception:
            st.markdown(f"""
            * **{t['about_meta_arch']}:** `SVM_Linear` (Support Vector Machine)
            * **{t['about_meta_features']}:** `31` features (5 bands + 26 vegetation indices)
            * **{t['about_meta_status']}:** Offline ⚠️
            * **{t['about_meta_region']}:** {t['about_meta_region_val']}
            """)
            
    st.write("---")
    st.markdown(f"<p style='text-align: center; color: #5a6b5d;'>{t['about_footer']}</p>", unsafe_allow_html=True)
