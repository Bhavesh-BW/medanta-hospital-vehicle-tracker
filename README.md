# Medanta Hospital - Vehicle Access Control & ANPR System

> High-accuracy Automatic Number Plate Recognition (ANPR) and hospital perimeter security system. Monitors vehicle entry/exit gates, calculates dwell and turnaround times, and enforces vehicle security watchlists.

---

## 🚀 Key Features
- **Two-stage pipeline**: YOLOv8n vehicle confirmation + YuNet license plate localization
- High-speed character recognition via PaddleOCR PP-OCRv3 + CRNN ONNX models
- Indian vehicle registration format validator & state code corrector (DL, HR, UP, MH, etc.)
- Automatic gate pairing (matches Entry and Exit passages for turnaround calculation)
- Live MJPEG video stream with bounding boxes and recognized plates
- Hourly traffic volume charts, peak hour analytics, and Excel report export
- Zero-setup bundled web dashboard (runs directly via Python backend)

---

## 🧠 Pre-Installed Models
The repository includes all required pre-trained weights and ONNX model files:
- `backend/yolov8n.onnx`: YOLOv8n vehicle detector
- `backend/.vision-models/license_plate_detection_lpd_yunet_2023mar.onnx`: Plate detector
- `backend/.vision-models/text_detection_en_ppocrv3_2023may.onnx`: Text region detector
- `backend/.vision-models/text_recognition_CRNN_EN_2023feb_fp16.onnx`: Character recognizer

---

## 🛠️ Quick Start Guide

### 1. Prerequisites
- **Python**: Version `3.9`, `3.10`, or `3.11` recommended.
- **Hardware**: Compatible with standard CPU. A CUDA-capable NVIDIA GPU is recommended for high-FPS multi-stream processing.
- **Node.js**: *(Optional)* The production UI is **pre-compiled** inside `backend/dist` and served automatically by the Python backend. Node.js is only required if you want to modify the React frontend.

---

### 2. Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Bhavesh-BW/medanta-hospital-vehicle-tracker.git
   cd medanta-hospital-vehicle-tracker
   ```

2. **Create and activate a virtual environment**:
   - **Windows**:
     ```bash
     python -m venv venv
     venv\Scripts\activate
     ```
   - **Linux / macOS**:
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

### 3. Running the Service

#### Option A: One-Click Launch Script
- **Windows**: Double-click `start.bat` (or run `start.bat` in CMD)
- **Linux / macOS**: Run `./start.sh`

#### Option B: Direct Python Execution
```bash
cd backend
python vehicle_tracker.py
```

---

### 4. Accessing the Dashboard

Once started, open your web browser and navigate to:
```
http://localhost:8176
```

---

## ⚙️ Configuration & Customization

The service can be customized using environment variables or directly in the UI:

| Variable | Default | Description |
| :--- | :--- | :--- |
| `TRACKER_BIND` | `0.0.0.0` | Host IP interface to bind the web server |
| `TRACKER_PORT` / `PORT` | `8176` | Port for HTTP API and Web Dashboard |
| `DEVICE` | Auto (`cuda:0` / `cpu`) | Inference device execution provider |

---

## 📁 Repository Structure
```
├── backend/
│   ├── vehicle_tracker.py      # Main engine and HTTP server
│   ├── dist/                 # Pre-built production web dashboard
│   ├── models/               # Neural network weights & ONNX models
│   └── requirements.txt      # Python dependencies
├── ui/                       # Frontend source code (React / Vite / Tailwind)
│   ├── src/                  # Components, hooks, and views
│   └── package.json          # Node.js dependencies (for UI development)
├── start.bat                 # Windows one-click launcher
├── start.sh                  # Linux / macOS one-click launcher
├── .gitignore                # Excludes node_modules, temp files & databases
└── README.md                 # Project documentation
```

---

## 🔒 Database & Privacy Notice
Local runtime databases (`*.db`), temporary snapshot photos, and security tokens are excluded from this repository via `.gitignore`. 
When running for the first time, SQLite database tables are automatically initialized with fresh schema structures.
