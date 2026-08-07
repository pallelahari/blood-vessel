# 🔬 Blood Vessel Segmentation & Analysis Platform

A full-stack biomedical web application for automated blood vessel segmentation, morphological quantification, and interactive annotation of microscopy images.

---

## 📋 Table of Contents

- [What It Does](#what-it-does)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation & Setup](#installation--setup)
- [Running the Application](#running-the-application)
- [How to Use](#how-to-use)
- [API Endpoints](#api-endpoints)
- [Output Files](#output-files)
- [Metrics Reference](#metrics-reference)
- [Tech Stack](#tech-stack)

---

## What It Does

Upload a microscopy image and its GeoJSON annotation file — the platform automatically:

1. Converts polygon annotations into a binary vessel mask
2. Skeletonizes the mask to extract vessel centerlines
3. Detects blobs (connected vessel components) and computes 18+ metrics per blob
4. Renders 5 analysis overlays viewable in the browser
5. Lets you hover over any vessel to inspect its metrics in real time
6. Lets you annotate branching points interactively and export enriched CSVs
7. Shows a **Dashboard** with summary stats and charts across all blobs

---

## Project Structure

```
blood-vessel-segmentation-main/
│
├── app.py                        ← Entry point — starts Flask on port 8000
│
├── backend/
│   ├── app.py                    ← REST API routes (Flask)
│   └── pipeline.py               ← Core processing: decode → mask → analyze → render
│
├── frontend/
│   ├── src/
│   │   ├── App.jsx               ← Full React UI (Viewer + Dashboard)
│   │   └── App.css               ← Dark theme stylesheet
│   ├── package.json
│   └── vite.config.js
│
├── vessel_quantification.py      ← Scientific algorithms (skeletonize, blob stats)
├── geojson_to_mask.py            ← GeoJSON polygon → binary mask converter
├── vessel_analysis.ipynb         ← Research/exploratory Jupyter notebook
├── requirements.txt              ← Python dependencies
└── README.md                     ← This file
```

---

## Prerequisites

Make sure these are installed before you begin:

| Tool | Version | Download |
|---|---|---|
| Python | 3.10 or later | https://python.org |
| Node.js | 18 LTS or later | https://nodejs.org |

> **Windows tip:** During Python install, check **"Add Python to PATH"**. For Node.js, use the **Windows Installer (.msi)** — ignore the Docker option.

---

## Installation & Setup

### 1. Extract the project

Unzip the project to a simple path like `C:\Projects\` or your Desktop. Avoid paths with spaces if possible.

### 2. Backend setup (run once)

Open Command Prompt inside the project root folder:

```bash
# Click the address bar in File Explorer, type cmd, press Enter

# Create a virtual environment
python -m venv venv

# Activate it (Windows)
venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt
```

### 3. Frontend setup (run once)

Open a **second** Command Prompt window:

```bash
cd frontend

# Install Node dependencies + charting library
npm install
npm install recharts
```

---

## Running the Application

You need **two terminals open at the same time** — one for the backend, one for the frontend.

### Terminal 1 — Backend

```bash
# From the project root folder
venv\Scripts\activate
python app.py
```

You should see:
```
* Running on http://127.0.0.1:8000
```

### Terminal 2 — Frontend

```bash
cd frontend
npm run dev
```

You should see:
```
➜  Local:   http://localhost:5173/
```

Open **http://localhost:5173** in your browser. The app is ready. ✅

> **Important:** Keep both terminal windows open while using the app. Closing either one will stop the app.

---

## How to Use

### Step 1 — Upload Files
In the sidebar, upload:
- **Image file** — microscopy image (`.png`, `.jpg`, `.tif`)
- **GeoJSON file** — vessel polygon annotations (`.geojson` or `.json`)

### Step 2 — Run Analysis
Click **▶ Run Analysis**. A live progress log appears in the center showing each processing step. Analysis time depends on image size.

### Step 3 — View Results
Use the 5 view buttons to switch between overlays:

| View | Description |
|---|---|
| **Original** | Raw grayscale input image |
| **Binary Mask** | White = vessel regions from GeoJSON annotations |
| **Blob Labels** | Each connected vessel blob colored and numbered |
| **Skeleton** | Vessel centerlines (green) drawn on the mask |
| **Morphology Map** | Blue = elongated, Red = compact, Pink = intermediate |

### Step 4 — Inspect Vessel Metrics
Hover your mouse over any vessel blob. A tooltip and full metrics panel appear showing all 18+ metrics for that blob in real time.

### Step 5 — View Dashboard
Click the **📊 Dashboard** tab in the top bar to see:
- 12 summary stat cards (total blobs, averages, morphology counts)
- Morphology pie chart
- Area and circularity histograms
- Length vs Width scatter plot (color coded by morphology)
- Top 20 blobs by skeleton length bar chart

### Step 6 — Annotate Branching Points *(optional)*
1. Click **⚪ Enable Branch Marking** in the sidebar
2. Click on vessel branch points in the image — red markers appear
3. Use **Undo / Redo / Clear** to adjust
4. Click **✓ Save Branching** — an enriched CSV is saved to the output folder and a green toast notification confirms the save

---

## API Endpoints

The frontend and backend communicate over HTTP. All endpoints are served at `http://127.0.0.1:8000`.

| Endpoint | Method | Description |
|---|---|---|
| `/api/process` | POST | Upload image + GeoJSON, run full analysis, return 5 images + blob data |
| `/api/progress` | GET | Returns live log messages during processing (polled every 1 second) |
| `/api/blob-info?x=&y=` | GET | Returns all metrics for the blob at pixel coordinate (x, y) |
| `/api/finalize-branches` | POST | Save user-marked branch points and export enriched CSV |
| `/api/health` | GET | Health check — returns `{"ok": true}` |

---

## Output Files

Each run creates a timestamped folder inside `analysis_outputs/` named after your image:

```
analysis_outputs/
└── <image_name>__20260615_142625/
    ├── original.png                       ← Grayscale input
    ├── binary_mask.png                    ← Vessel mask
    ├── blob_labeled.png                   ← Color-coded blobs
    ├── skeleton_overlay.png               ← Skeleton on mask
    ├── compact_overlay.png                ← Morphology color map
    ├── blob_metrics.csv                   ← All metrics for every blob
    ├── blob_metrics_with_branching.csv    ← Enriched CSV (after Save Branching)
    ├── input.geojson                      ← Copy of the uploaded GeoJSON
    ├── intermediates.npz                  ← NumPy arrays (mask, labels, skeleton)
    └── branch_points.json                 ← Saved branch point coordinates
```

Multiple runs of the same image create separate folders — they never overwrite each other.

---

## Metrics Reference

Every vessel blob gets these metrics computed and saved in the CSV:

| Metric | Description |
|---|---|
| `blob_id` | Unique integer ID per connected vessel component |
| `morphology_2d` | Classification: `compact` / `elongated` / `intermediate` / `degenerate` |
| `circularity_2d` | 4π·area/perimeter² — 1.0 = perfect circle, lower = more elongated |
| `eccentricity_2d` | Best-fit ellipse eccentricity — 0 = circle, 1 = line |
| `area_px` | Total foreground pixel count |
| `n_segments` | Number of skeleton branches within the blob |
| `total_skeleton_length_px` | Sum of all skeleton segment lengths |
| `max_segment_length_px` | Length of the longest single skeleton segment |
| `centerline_network_diameter_px` | Longest shortest path through the skeleton graph |
| `centerline_network_diameter_unmerged_px` | Network diameter on raw unmerged segments |
| `centerline_network_diameter_merged_px` | Network diameter after merging co-linear segments |
| `width_mean_length_weighted_px` | Mean vessel width weighted by segment length |
| `width_mean_over_segments_px` | Simple mean width averaged equally over all segments |
| `width_along_longest_path_px` | Mean width along the longest skeleton path |
| `width_along_longest_unmerged_path_px` | Width along the longest unmerged path |
| `width_along_longest_merged_path_px` | Width along the longest merged path |
| `length_clean` | Cleaned length — for compact blobs with no skeleton, estimated as √(area/π) |
| `width_clean` | Cleaned width — falls back to length value when skeleton width is zero |

---

## Tech Stack

### Backend
| Library | Purpose |
|---|---|
| `Flask` | Lightweight Python web framework — serves the REST API |
| `flask-cors` | Allows the browser (port 5173) to call the backend (port 8000) |
| `opencv-python-headless` | Image decode, resize, color conversion, PNG encoding |
| `scikit-image` | Skeletonization, connected component labeling |
| `scipy` | Morphological operations |
| `numpy` | All array mathematics |
| `pandas` | Blob metrics table and CSV export |

### Frontend
| Library | Purpose |
|---|---|
| `React 18` | UI component framework with reactive state |
| `Vite` | Fast dev server and build tool |
| `recharts` | Charts library — bar, pie, scatter charts for the dashboard |

---

## Troubleshooting

**App stuck on "Processing…" for a long time**
> Large high-resolution images take longer to skeletonize. Check the backend terminal — it prints live progress logs. If it appears frozen, restart the backend and try again.

**`npm run dev` fails**
> Make sure you ran `npm install` and `npm install recharts` inside the `frontend/` folder first.

**`pip install` fails**
> Make sure your virtual environment is activated — you should see `(venv)` at the start of the command line.

**Tooltip shows no data on hover**
> Make sure the backend is still running in Terminal 1. Try restarting it with `python app.py`.

**Port already in use error**
> Another process is using port 8000 or 5173. Restart your computer or find and close the process using that port.
