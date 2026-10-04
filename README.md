# Border Surveillance System — Python Edition
### Sentinel-1 / Sentinel-2 · Change Detection · ML/DL Algorithm Comparison

---

## Project Structure

```
bss-python/
├── app.py               ← Streamlit web UI  (run this for the browser interface)
├── main.py              ← CLI entry point   (run this for terminal use)
├── algorithms.py        ← All 5 ML/DL algorithms  ← NEW FILE
├── change_detection.py  ← Dispatcher: pixel methods + ML algorithms
├── sentinel_fetch.py    ← Copernicus STAC API query + image download
├── visualise.py         ← Matplotlib 4-panel report generator
├── requirements.txt     ← Python dependencies
└── README.md            ← This file
```

---

## VS Code Setup on macOS

### Step 1 — Check Python
```bash
python3 --version   # need 3.10+
```

### Step 2 — Open project
File → Open Folder… → select the `bss-python` folder → Open

### Step 3 — Install Python extension
Cmd + Shift + X → search "Python" by Microsoft → Install

### Step 4 — Open Terminal
Terminal → New Terminal  (Ctrl + `)

### Step 5 — Virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 6 — Install dependencies
```bash
pip install -r requirements.txt
```

---

## Running the Streamlit UI

```bash
streamlit run app.py
```

Opens in your browser at http://localhost:8501

### UI Features
- **Border Region Preset** dropdown (Jammu, Kabul, Bhutan, etc.) or custom lat/lon
- **Satellite** selector — Sentinel-2 optical or Sentinel-1 SAR
- **Algorithm** dropdown — choose from all 8 methods (see table below)
- **Threshold** slider — sensitivity of change detection
- **Download** button — save the report PNG

---

## Running from the Terminal

```bash
# Default (MAD pixel method)
python main.py --lat 32.72 --lon 74.86

# ML algorithms
python main.py --preset jammu --algo svm
python main.py --preset jammu --algo kmeans
python main.py --preset jammu --algo random_forest
python main.py --preset jammu --algo siamese
python main.py --preset jammu --algo alphaearth

# Classic pixel methods
python main.py --preset jammu --algo mad
python main.py --preset jammu --algo ratio  --satellite S1
python main.py --preset bhutan --algo ndvi

# Open report immediately
python main.py --preset jammu --algo random_forest --show
```

---

## All Algorithms

| Flag | Display Name | Type | How It Works | Best For |
|---|---|---|---|---|
| `mad` | Mean Absolute Difference | Pixel | Mean RGB delta per pixel | Fast baseline |
| `ratio` | Log-Ratio | Pixel | Log-scale ratio | SAR imagery |
| `ndvi` | NDVI Change | Pixel | Vegetation index diff | Forest/agriculture |
| `kmeans` | K-Means Clustering | ML Unsupervised | 5-cluster ranking by change intensity | Exploratory |
| `svm` | Support Vector Machine | ML Supervised | RBF-kernel SVC on 13-D features | Sharp boundaries |
| `random_forest` | Random Forest | ML Supervised | 80 trees, 22-D features, P(change) | Probabilistic maps |
| `siamese` | Siamese Network | Deep Learning | Shared encoder → L2 feature distance | Illumination-robust |
| `alphaearth` | AlphaEarth Foundation | Foundation Model | ViT patch embed → cosine distance | Multi-season/sensor |

---

## All CLI Options

```
--lat           Latitude  (decimal °N)
--lon           Longitude (decimal °E)
--preset        jammu | kabul | pak-loc | bhutan | bangladesh | kutch
--satellite     S2 (optical, default) or S1 (SAR)
--radius        Search radius in degrees  (default: 0.2)
--cloud         Max cloud cover % for S2  (default: 20)
--days-back     Days back to search  (default: 180)
--algo          mad | ratio | ndvi | kmeans | svm | random_forest | siamese | alphaearth
--threshold     Change magnitude threshold 0-255 (default: 40)
--seed          Random seed for ML algorithms (default: 42)
--output        Output PNG path (default: bss_report.png)
--show          Open the PNG after saving
```

---

## How Data Works

The system queries the free Copernicus STAC API (no key needed):
```
https://earth-search.aws.element84.com/v1/search
```
If offline or no scenes match, it falls back to procedurally generated
placeholder imagery so the pipeline always produces output.
# Border_Surveillance_Systems
