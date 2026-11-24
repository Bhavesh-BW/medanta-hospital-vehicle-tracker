"""Vehicle Tracker — continuous backend service for end-to-end vehicle journeys.
Runs on the GPU host next to objects-gpu/anpr-gpu. Owns:
  - device registry (camera stream + entry/exit/checkpoint role + ANPR toggle)
  - journey rules (visit gap, exit grace, passage correlation)
  - a continuous multi-stream ANPR collector (PP-OCRv3 + CRNN, same models as anpr-gpu)
  - vehicle_passage polling from the objects-gpu services for checkpoint cameras
  - per-plate journey computation: entry -> cameras seen -> dwell -> exit
Storage: SQLite at DB_PATH (mount a volume at /data).
Auth: X-Local-Vision-Key header, same shared key as the other vision services.
"""
import base64
import collections
import hmac
import json
import os
import pathlib
import re
import sqlite3
import threading
import time
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs
os.environ.setdefault('OPENCV_FFMPEG_CAPTURE_OPTIONS', 'rtsp_transport;tcp')
import cv2
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parent
DIST_DIR = ROOT / 'dist'
KEY = os.environ.get('VISION_SERVICE_KEY', '')
MODELS = ROOT / '.vision-models'
import sys
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(MODELS))
try:
    from lpd_yunet import LPD_YuNet
except Exception:
    LPD_YuNet = None
LIVE_DB_PATH = os.environ.get('TRACKER_DB', str(ROOT / 'vehicle_tracker.db') if (os.name == 'nt' or not os.path.exists('/data')) else '/data/vehicle_tracker.db')
DEMO_DB_PATH = '/data/vehicle_tracker_demo.db' if os.path.exists('/data') else str(ROOT / 'vehicle_tracker_demo.db')
RECORDINGS_DIR = pathlib.Path('/data/recordings') if os.path.exists('/data/recordings') else (ROOT / 'recordings' if (ROOT / 'recordings').exists() else pathlib.Path(r"d:\Python Projects\recordings"))

def get_playlist(path_or_dir):
    if not path_or_dir:
        return []
    p = str(path_or_dir)
    if os.path.isdir(p):
        files = sorted([
            os.path.join(p, f) for f in os.listdir(p)
            if f.lower().endswith(('.mp4', '.avi', '.mkv', '.mov', '.ts'))
        ])
        if files:
            return files
    if not os.path.exists(p) and os.path.isdir(os.path.dirname(p)):
        parent = os.path.dirname(p)
        prefix = os.path.basename(p)
        matched = sorted([
            os.path.join(parent, f) for f in os.listdir(parent)
            if f.lower().startswith(prefix.lower()) and f.lower().endswith(('.mp4', '.avi', '.mkv', '.mov', '.ts'))
        ])
        if matched:
            return matched
    if os.path.isfile(p):
        return [p]
    return [p]

DEMO_VEHICLE_RECORDINGS = {
    'mgptz': str(RECORDINGS_DIR / "mgptz"),
    'mg_outside': str(RECORDINGS_DIR / "mg_outside"),
    'MgOutside': str(RECORDINGS_DIR / "mg_outside"),
    'exit_gate': str(RECORDINGS_DIR / "exit_gate"),
    'ExitGate': str(RECORDINGS_DIR / "exit_gate"),
    'egtmp': str(RECORDINGS_DIR / "egtmp"),
    'EgToMp': str(RECORDINGS_DIR / "egtmp"),
    'basement': str(RECORDINGS_DIR / "basement"),
    'basement_entry': str(RECORDINGS_DIR / "basement"),
}
DEFAULT_VEHICLE_DEVICES = [
    {'stream': 'mgptz', 'name': 'Main Gate PTZ (Entry)', 'role': 'entry', 'anpr': 1, 'url': ''},
    {'stream': 'mg_outside', 'name': 'Main Gate Outside (Inbound)', 'role': 'entry', 'anpr': 1, 'url': ''},
    {'stream': 'exit_gate', 'name': 'Exit Gate (Outbound)', 'role': 'exit', 'anpr': 1, 'url': ''},
    {'stream': 'egtmp', 'name': 'Exit Gate to Main Porch', 'role': 'checkpoint', 'anpr': 1, 'url': ''},
]
MODE_STATE_FILE = ROOT / "feed_mode.json"
CURRENT_FEED_MODE = 'live'
if os.path.exists(MODE_STATE_FILE):
    try:
        with open(MODE_STATE_FILE, 'r') as fp:
            CURRENT_FEED_MODE = json.load(fp).get('mode', 'live')
    except Exception:
        pass

DB_PATH = DEMO_DB_PATH if CURRENT_FEED_MODE == 'demo' else LIVE_DB_PATH
RTSP_BASE = os.environ.get('RTSP_BASE', 'rtsp://124.123.89.162:31554/live')
OBJECTS_URLS = [u.strip() for u in os.environ.get('OBJECTS_URLS', 'http://172.17.0.1:31012,http://172.17.0.1:31013').split(',') if u.strip()]

# IST Timezone helper (+05:30)
IST_OFFSET_SEC = int(5.5 * 3600)
IST_OFFSET_MS = int(5.5 * 3600 * 1000)

def to_ist_strftime(epoch_sec=None, fmt='%Y-%m-%d %H:%M:%S'):
    sec = epoch_sec if epoch_sec is not None else time.time()
    return time.strftime(fmt, time.gmtime(sec + IST_OFFSET_SEC))
STREAM_RE = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_-]{0,40}$')
ROLES = ('entry', 'exit', 'checkpoint')
MIME_TYPES = {
    '.html': 'text/html; charset=utf-8',
    '.js': 'application/javascript; charset=utf-8',
    '.css': 'text/css; charset=utf-8',
    '.json': 'application/json',
    '.png': 'image/png',
    '.jpg': 'image/jpeg',
    '.jpeg': 'image/jpeg',
    '.svg': 'image/svg+xml',
    '.ico': 'image/x-icon',
    '.woff': 'font/woff',
    '.woff2': 'font/woff2',
}
ort = None
try:
    if int(np.__version__.split('.')[0]) < 2:
        import onnxruntime as ort
except Exception:
    ort = None
CHARSET = '0123456789abcdefghijklmnopqrstuvwxyz'
STRICT_RE = re.compile(r'^[a-z]{2}\d{1,2}[a-z]{1,3}\d{4}$')
TO_DIGIT = {'o': '0', 'q': '0', 'd': '0', 'i': '1', 'l': '1', 'j': '1', 'z': '2', 'e': '3', 'a': '4', 's': '5', 'g': '6', 't': '7', 'b': '8',
            'O': '0', 'Q': '0', 'D': '0', 'I': '1', 'L': '1', 'J': '1', 'Z': '2', 'E': '3', 'A': '4', 'S': '5', 'G': '6', 'T': '7', 'B': '8'}
TO_LETTER = {'0': 'O', '1': 'I', '2': 'Z', '3': 'E', '4': 'A', '5': 'S', '6': 'G', '7': 'T', '8': 'B'}
INDIAN_STATES = {'HR', 'DL', 'UP', 'MH', 'KA', 'PB', 'CH', 'UK', 'RJ', 'MP', 'GJ', 'WB', 'TN', 'KL', 'AP', 'TS', 'BR', 'JH', 'OD', 'AS', 'GA', 'JK', 'HP', 'TR', 'NL', 'MN', 'MZ', 'SK', 'BH'}
PRIORITY_STATES = ['HR', 'DL', 'UP', 'PB', 'CH', 'RJ', 'UK', 'MH']
COMMON_PREFIX_FIXES = {
    '11R': 'HR', 'IIR': 'HR', 'MR': 'HR', 'NR': 'HR', 'HB': 'HR', '1R': 'HR', 'IR': 'HR',
    '0L': 'DL', 'D1': 'DL', 'DI': 'DL', 'OL': 'DL', 'QL': 'DL', 'CL': 'DL',
    'VP': 'UP', 'OP': 'UP', 'LP': 'UP', '0P': 'UP', 'JP': 'UP',
    'P8': 'PB', 'R1': 'RJ', 'C11': 'CH'
}
GARBAGE_WORDS = [
    'exit', 'gate', 'entry', 'main', 'medanta', 'app', 'hospital',
    'outside', 'ptz', 'egtmp', 'speed', 'slow', 'stop', 'police',
    'traffic', 'parking', 'park', 'camera', 'caution', 'notice',
    'live', 'mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun',
    'kalka', 'gurgaon', 'delhi', 'porch', 'tower', 'block'
]
def is_garbage_text(raw):
    if not raw:
        return True
    t = raw.lower()
    return any(bad in t for bad in GARBAGE_WORDS)
def is_watermark_zone(cx, cy, w, h):
    # Top 11% (timestamp: e.g. "11-12-2026 Thu 15:23:44")
    if cy < int(h * 0.11):
        return True
    # Bottom 9% across entire frame
    if cy > int(h * 0.91):
        return True
    # Bottom-right corner (where "Exit gate", "Main Gate Outside" is stamped)
    if cx > int(w * 0.45) and cy > int(h * 0.78):
        return True
    # Bottom-left corner
    if cx < int(w * 0.25) and cy > int(h * 0.85):
        return True
    return False
def clean_indian_text(raw):
    if not raw:
        return ""
    t = re.sub(r'[^A-Za-z0-9]', '', raw).upper()
    if t.startswith('IND') and len(t) >= 11:
        t = t[3:]
    elif (t.startswith('I') or t.startswith('1')) and len(t) >= 10 and t[1:3] in INDIAN_STATES:
        t = t[1:]
    for pfx, fixed in COMMON_PREFIX_FIXES.items():
        if t.startswith(pfx):
            t = fixed + t[len(pfx):]
            break
    return t
def fix_plate(raw_text):
    if is_garbage_text(raw_text):
        return None
    text = clean_indian_text(raw_text)
    if is_garbage_text(text) or len(text) < 7 or len(text) > 11:
        return None

    # First 2 chars must be State letters
    s0 = TO_LETTER.get(text[0], text[0])
    s1 = TO_LETTER.get(text[1], text[1])
    state = s0 + s1
    if state not in INDIAN_STATES:
        found_state = None
        for s in PRIORITY_STATES:
            if s[0] == s0 or s[1] == s1:
                found_state = s
                break
        if found_state:
            state = found_state
        else:
            return None

    # Last 4 characters must be digits
    tail_chars = text[-4:]
    fixed_tail = [ch if ch.isdigit() else TO_DIGIT.get(ch) for ch in tail_chars]
    if any(c is None for c in fixed_tail):
        return None
    fixed_tail_str = ''.join(fixed_tail)

    mid = text[2:-4]
    if not mid:
        return None

    # Determine RTO digits and Series letters
    rto = []
    if state == 'DL':
        rto.append(mid[0] if mid[0].isdigit() else TO_DIGIT.get(mid[0], '1'))
        start_idx = 1
        if len(mid) >= 2 and (mid[1].isdigit() or (mid[1] in TO_DIGIT and len(mid) >= 3 and mid[2].isalpha())):
            rto.append(mid[1] if mid[1].isdigit() else TO_DIGIT.get(mid[1], '0'))
            start_idx = 2
    else:
        d1 = mid[0] if mid[0].isdigit() else TO_DIGIT.get(mid[0], '2')
        rto.append(d1)
        start_idx = 1
        if len(mid) >= 2:
            d2 = mid[1] if mid[1].isdigit() else TO_DIGIT.get(mid[1], '6')
            rto.append(d2)
            start_idx = 2

    series = [ch if ch.isalpha() else TO_LETTER.get(ch, 'A') for ch in mid[start_idx:]]
    candidate = state + ''.join(rto) + ''.join(series) + fixed_tail_str
    if 8 <= len(candidate) <= 10:
        return candidate
    return None
cv2.setNumThreads(4)
class VehicleDetector:
    """YOLO vehicle & license plate detector with dual-engine (ONNXRuntime / OpenCV-DNN)
    and automatic custom model detection (vehicle.onnx vs yolov8n.onnx)."""
    def __init__(self):
        candidate_paths = [
            ROOT / 'vehicle.onnx',
            pathlib.Path('/app/vehicle.onnx'),
            MODELS / 'vehicle.onnx',
            pathlib.Path('/app/yolov8n.onnx'),
            ROOT / 'yolov8n.onnx',
            MODELS / 'yolov8n.onnx',
            pathlib.Path('/app/.vision-models/yolov8n.onnx')
        ]
        model_path = next((p for p in candidate_paths if p.is_file()), None)
        if not model_path:
            raise FileNotFoundError("No vehicle model found (checked vehicle.onnx and yolov8n.onnx)")
        self.model_path = model_path
        self.use_ort = False
        self.sess = None
        self.net = None

        if ort is not None:
            try:
                opts = ort.SessionOptions()
                opts.log_severity_level = 3
                avail = ort.get_available_providers()
                providers = [p for p in ["CUDAExecutionProvider", "CPUExecutionProvider"] if p in avail]
                self.sess = ort.InferenceSession(str(model_path), opts, providers=providers)
                self.input_name = self.sess.get_inputs()[0].name
                self.output_name = self.sess.get_outputs()[0].name
                dummy = np.zeros((1, 3, 640, 640), dtype=np.float32)
                for _ in range(2):
                    self.sess.run([self.output_name], {self.input_name: dummy})
                self.use_ort = True
            except Exception:
                self.use_ort = False

        if not self.use_ort:
            self.net = cv2.dnn.readNet(str(model_path))
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)

        # Detect model type:
        # Custom 5-class model: 0: car, 1: no_plate, 2: auto, 3: bike, 4: ambulance
        # COCO 80-class fallback: 2: car, 3: motorcycle, 5: bus, 7: truck
        self.is_custom = ('vehicle' in model_path.name.lower() and 'yolov8n' not in model_path.name.lower())
        if self.is_custom:
            self.cls_names = {0: 'car', 1: 'no_plate', 2: 'auto', 3: 'bike', 4: 'ambulance'}
            self.v_cls = [0, 1, 2, 3, 4]
        else:
            self.cls_names = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
            self.v_cls = [2, 3, 5, 7]

        print(f"[VehicleDetector] Successfully loaded: {model_path.name} (Engine: {'ONNXRuntime' if self.use_ort else 'OpenCV-DNN'})")

    def detect(self, frame, conf_thresh=0.25):
        h, w = frame.shape[:2]
        if self.use_ort:
            img = cv2.resize(frame, (640, 640)).astype(np.float32) / 255.0
            img = np.transpose(img, (2, 0, 1))[None, ...]
            preds = self.sess.run([self.output_name], {self.input_name: img})[0]
            out = preds[0].T
        else:
            blob = cv2.dnn.blobFromImage(frame, 1/255.0, (640, 640), swapRB=True, crop=False)
            self.net.setInput(blob)
            preds = self.net.forward()
            out = preds[0].T

        boxes = out[:, :4]
        scores = out[:, 4:]

        v_cls = self.v_cls
        if scores.shape[1] == 5 and not self.is_custom:
            v_cls = [0, 1, 2, 3, 4]
            self.cls_names = {0: 'car', 1: 'no_plate', 2: 'auto', 3: 'bike', 4: 'ambulance'}
        elif scores.shape[1] > 10 and self.is_custom:
            v_cls = [2, 3, 5, 7]
            self.cls_names = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}

        max_scores = np.max(scores[:, v_cls], axis=1)
        cls_idx = np.argmax(scores[:, v_cls], axis=1)
        mask = max_scores >= conf_thresh
        if not np.any(mask):
            return []

        v_boxes = boxes[mask]
        v_scores = max_scores[mask]
        v_cls_chosen = [v_cls[i] for i in cls_idx[mask]]
        scale_x, scale_y = w / 640.0, h / 640.0

        candidate_boxes = []
        scores_list = []
        for b, s, c in zip(v_boxes, v_scores, v_cls_chosen):
            cx, cy, bw, bh = b
            x1 = max(0, int((cx - bw / 2) * scale_x))
            y1 = max(0, int((cy - bh / 2) * scale_y))
            x2 = min(w, int((cx + bw / 2) * scale_x))
            y2 = min(h, int((cy + bh / 2) * scale_y))
            candidate_boxes.append([x1, y1, x2 - x1, y2 - y1])
            scores_list.append(float(s))

        indices = cv2.dnn.NMSBoxes(candidate_boxes, scores_list, conf_thresh, 0.45)
        results = []
        for i in indices:
            idx = i[0] if isinstance(i, (list, tuple, np.ndarray)) else i
            bx, by, bw, bh = candidate_boxes[idx]
            cid = v_cls_chosen[idx]
            results.append({
                "bbox": [bx, by, bx + bw, by + bh],
                "score": round(scores_list[idx], 2),
                "cls": cid,
                "name": self.cls_names.get(cid, 'vehicle')
            })
        return results
class Detector:
    """Multi-stage Hybrid ANPR Pipeline:
       Stage 1: YOLOv8 confirms vehicle presence.
       Stage 2A: Direct plates from custom YOLO vehicle model (class 'no_plate').
       Stage 2B: Dedicated YuNet License Plate Detector (LPD) with 4-corner perspective unwarping.
       Stage 2C: DBNet text region fallback if YuNet has low confidence.
       Stage 3: High-resolution PP-OCRv4 recognition with 2-line stacked plate support & character normalization."""
    def __init__(self):
        self.vehicle_detector = VehicleDetector()

        # 1. DBNet text detector (fallback)
        self.detector = cv2.dnn_TextDetectionModel_DB(cv2.dnn.readNet(str(MODELS / 'text_detection_en_ppocrv3_2023may.onnx')))
        self.detector.setBinaryThreshold(0.25)
        self.detector.setPolygonThreshold(0.40)
        self.detector.setUnclipRatio(1.8)
        self.detector.setMaxCandidates(200)
        self.detector.setInputSize((736, 736))
        self.detector.setInputMean((123.675, 116.28, 103.53))
        self.detector.setInputScale(1.0 / 255.0 / np.array([0.229, 0.224, 0.225]))

        # 2. Dedicated YuNet License Plate Detector
        yunet_path = MODELS / 'license_plate_detection_lpd_yunet_2023mar.onnx'
        self.plate_detector = None
        if LPD_YuNet is not None and yunet_path.is_file():
            try:
                self.plate_detector = LPD_YuNet(modelPath=str(yunet_path), inputSize=[320, 240], confThreshold=0.22)
                print("[Detector] Successfully loaded YuNet License Plate Detector")
            except Exception as e:
                print(f"[Detector] YuNet init notice: {e}")

        # 3. High-Accuracy PP-OCRv4 Character Recognizer
        ppocr_path = MODELS / 'en_PP-OCRv4_rec.onnx'
        dict_path = MODELS / 'en_dict.txt'
        self.use_ppocrv4 = False
        if ppocr_path.is_file() and dict_path.is_file():
            try:
                with open(dict_path, 'r', encoding='utf-8') as f:
                    lines = [line.strip('\r\n') for line in f.readlines()]
                self.ppocr_chars = ['blank'] + lines + [' ']
                self.recognizer_v4 = cv2.dnn.readNet(str(ppocr_path))
                self.use_ppocrv4 = True
                print("[Detector] Successfully loaded PP-OCRv4 recognition engine (SVTR-LCNet)")
            except Exception as e:
                print(f"[Detector] PP-OCRv4 load error: {e}")

        # Fallback CRNN recognizer
        crnn_path = MODELS / 'text_recognition_CRNN_EN_2023feb_fp16.onnx'
        self.recognizer_crnn = cv2.dnn.readNet(str(crnn_path)) if crnn_path.is_file() else None

    @staticmethod
    def timestamp_mask(frame):
        mask = np.zeros(frame.shape[:2], dtype=np.uint8) + 255
        mask[:int(frame.shape[0] * 0.12), :] = 0
        return mask

    def _decode_ppocr(self, out):
        preds = out[0]
        text, confs = [], []
        last_idx = 0
        for t in range(preds.shape[0]):
            idx = int(np.argmax(preds[t]))
            score = float(preds[t, idx])
            if idx != 0 and idx != last_idx:
                text.append(self.ppocr_chars[idx])
                confs.append(score)
            last_idx = idx
        avg_conf = float(np.mean(confs)) if confs else 0.0
        return ''.join(text).upper().replace(' ', ''), avg_conf

    def ocr(self, image, quad):
        pts = quad.astype(np.float32).reshape(4, 2)
        w1 = np.linalg.norm(pts[1] - pts[0])
        w2 = np.linalg.norm(pts[2] - pts[3])
        h1 = np.linalg.norm(pts[3] - pts[0])
        h2 = np.linalg.norm(pts[2] - pts[1])
        pw = max(int(max(w1, w2)), 20)
        ph = max(int(max(h1, h2)), 10)

        dst = np.array([[0, 0], [pw - 1, 0], [pw - 1, ph - 1], [0, ph - 1]], dtype=np.float32)
        matrix = cv2.getPerspectiveTransform(pts, dst)
        crop = cv2.warpPerspective(image, matrix, (pw, ph))
        if crop is None or crop.size == 0:
            return ""

        if self.use_ppocrv4:
            aspect = pw / max(ph, 1)
            # 2-line plate handling (aspect ratio <= 2.2 and sufficient height)
            if aspect <= 2.2 and ph >= 26:
                mid_y = int(ph * 0.52)
                top = crop[:mid_y, :]
                bot = crop[mid_y:, :]

                top_res = cv2.resize(top, (320, 48))
                blob_top = cv2.dnn.blobFromImage(top_res, 1.0/127.5, (320, 48), (127.5, 127.5, 127.5), swapRB=True)
                self.recognizer_v4.setInput(blob_top)
                txt_top, _ = self._decode_ppocr(self.recognizer_v4.forward())

                bot_res = cv2.resize(bot, (320, 48))
                blob_bot = cv2.dnn.blobFromImage(bot_res, 1.0/127.5, (320, 48), (127.5, 127.5, 127.5), swapRB=True)
                self.recognizer_v4.setInput(blob_bot)
                txt_bot, _ = self._decode_ppocr(self.recognizer_v4.forward())
                return txt_top + txt_bot
            else:
                resized = cv2.resize(crop, (320, 48))
                blob = cv2.dnn.blobFromImage(resized, 1.0/127.5, (320, 48), (127.5, 127.5, 127.5), swapRB=True)
                self.recognizer_v4.setInput(blob)
                text, _ = self._decode_ppocr(self.recognizer_v4.forward())
                return text
        elif self.recognizer_crnn is not None:
            gray = cv2.cvtColor(cv2.resize(crop, (100, 32)), cv2.COLOR_BGR2GRAY)
            self.recognizer_crnn.setInput(cv2.dnn.blobFromImage(gray, size=(100, 32), mean=127.5, scalefactor=1 / 127.5))
            out = self.recognizer_crnn.forward()
            text = ''
            for index in range(out.shape[0]):
                best = int(np.argmax(out[index][0]))
                text += '-' if best == 0 else CHARSET[best - 1]
            return ''.join(char for i, char in enumerate(text) if char != '-' and (i == 0 or char != text[i - 1]))
        return ""

    def scan(self, frame):
        h, w = frame.shape[:2]
        # ── STAGE 1: Confirm Vehicle Presence First (CUDA YOLOv8 ~4.5ms) ─
        raw_detections = self.vehicle_detector.detect(frame, conf_thresh=0.25)
        if not raw_detections:
            return [], []

        vehicles = [d for d in raw_detections if d['name'] != 'no_plate']
        direct_plates = [d for d in raw_detections if d['name'] == 'no_plate']
        if not vehicles and not direct_plates:
            return [], []

        mask = self.timestamp_mask(frame)
        detections = []
        seen_quads = []

        # ── STAGE 1B: Direct License Plate Extraction (from custom vehicle.onnx) ──
        for p in direct_plates:
            px1, py1, px2, py2 = p['bbox']
            bw = px2 - px1
            bh = py2 - py1
            if bw < 20 or bh < 8 or bw > 600 or bh > 200:
                continue
            center = (px1 + bw / 2, py1 + bh / 2)
            cx, cy = int(center[0]), int(center[1])
            if is_watermark_zone(cx, cy, w, h):
                continue
            matching_v = None
            for v in vehicles:
                vx1, vy1, vx2, vy2 = v['bbox']
                if (vx1 - 25) <= cx <= (vx2 + 25) and (vy1 - 25) <= cy <= (vy2 + 25):
                    matching_v = v
                    break
            quad = np.array([[px1, py1], [px2, py1], [px2, py2], [px1, py2]], dtype=np.float64)
            raw_text = self.ocr(frame, quad)
            if not raw_text or len(raw_text) < 4 or is_garbage_text(raw_text):
                continue
            normalized = fix_plate(raw_text)
            if not normalized:
                continue
            seen_quads.append(center)
            detections.append({
                'plate': normalized.upper(),
                'quad': quad,
                'bbox': [px1, py1, bw, bh],
                'score': round(float(p['score']), 3),
                'vehicle_bbox': matching_v['bbox'] if matching_v else p['bbox'],
                'vehicle_name': matching_v['name'] if matching_v else 'vehicle'
            })

        # ── STAGE 2: Targeted Vehicle Crop ANPR (YuNet LPD + DBNet Fallback) ──
        def vehicle_area(v):
            vx1, vy1, vx2, vy2 = v['bbox']
            return (vx2 - vx1) * (vy2 - vy1)

        sorted_vehicles = sorted(vehicles, key=vehicle_area, reverse=True)[:2]

        for v in sorted_vehicles:
            vx1, vy1, vx2, vy2 = v['bbox']
            vw = vx2 - vx1
            vh = vy2 - vy1
            if vw < 50 or vh < 50:
                continue

            pad_x = int(vw * 0.10)
            pad_y = int(vh * 0.12)
            x0 = max(0, vx1 - pad_x)
            y0 = max(0, vy1 - pad_y)
            x1 = min(w, vx2 + pad_x)
            y1 = min(h, vy2 + pad_y)

            crop_w = x1 - x0
            crop_h = y1 - y0
            if crop_w < 40 or crop_h < 40:
                continue

            tile = frame[y0:y1, x0:x1].copy()
            tile_mask = mask[y0:y1, x0:x1]
            tile[tile_mask == 0] = 128

            vehicle_plate_found = False

            # Step 2A: Dedicated YuNet License Plate Detector
            if self.plate_detector is not None:
                try:
                    self.plate_detector.setInputSize([320, 240])
                    res_yunet = self.plate_detector.infer(cv2.resize(tile, (320, 240)))
                    for yr in res_yunet:
                        pts_320 = yr[:8].reshape(4, 2)
                        yscore = float(yr[8])
                        pts_tile = pts_320.copy()
                        pts_tile[:, 0] = pts_tile[:, 0] * (crop_w / 320.0) + x0
                        pts_tile[:, 1] = pts_tile[:, 1] * (crop_h / 240.0) + y0

                        center = pts_tile.mean(axis=0)
                        cx, cy = int(center[0]), int(center[1])
                        if is_watermark_zone(cx, cy, w, h):
                            continue
                        if any(np.hypot(center[0] - sc[0], center[1] - sc[1]) < 25 for sc in seen_quads):
                            continue

                        raw_text = self.ocr(frame, pts_tile)
                        normalized = fix_plate(raw_text)
                        if normalized:
                            seen_quads.append(center)
                            x_b, y_b, w_b, h_b = cv2.boundingRect(pts_tile.astype(np.int32))
                            detections.append({
                                'plate': normalized.upper(),
                                'quad': pts_tile,
                                'bbox': [x_b, y_b, w_b, h_b],
                                'score': round(yscore, 3),
                                'vehicle_bbox': v['bbox'],
                                'vehicle_name': v['name']
                            })
                            vehicle_plate_found = True
                except Exception:
                    pass

            # Step 2B: Fallback to DBNet if YuNet didn't find plate
            if not vehicle_plate_found:
                try:
                    boxes, scores = self.detector.detect(cv2.resize(tile, (736, 736)))
                except Exception:
                    continue
                if boxes is None:
                    continue

                sx, sy = crop_w / 736.0, crop_h / 736.0
                for box, score in zip(boxes, scores):
                    quad = np.array(box, dtype=np.float64).reshape(-1, 2)
                    quad[:, 0] = quad[:, 0] * sx + x0
                    quad[:, 1] = quad[:, 1] * sy + y0
                    x, y, bw, bh = cv2.boundingRect(quad.astype(np.int32))
                    if bw < 25 or bh < 8 or bw > 600 or bh > 180 or bw / max(bh, 1) < 1.3 or bw / max(bh, 1) > 9:
                        continue
                    center = (x + bw / 2, y + bh / 2)
                    cx, cy = int(center[0]), int(center[1])
                    if is_watermark_zone(cx, cy, w, h):
                        continue
                    pad_w_chk = vw * 0.15
                    pad_h_chk = vh * 0.15
                    if not ((vx1 - pad_w_chk) <= cx <= (vx2 + pad_w_chk) and (vy1 - pad_h_chk) <= cy <= (vy2 + pad_h_chk)):
                        continue
                    if any(np.hypot(center[0] - sc[0], center[1] - sc[1]) < 25 for sc in seen_quads):
                        continue
                    if cy < mask.shape[0] and cx < mask.shape[1] and mask[cy, cx] == 0:
                        continue
                    raw_text = self.ocr(frame, quad)
                    if not raw_text or len(raw_text) < 4 or is_garbage_text(raw_text):
                        continue
                    normalized = fix_plate(raw_text)
                    if not normalized:
                        continue
                    seen_quads.append(center)
                    detections.append({
                        'plate': normalized.upper(),
                        'quad': quad,
                        'bbox': [x, y, bw, bh],
                        'score': round(float(score), 3),
                        'vehicle_bbox': v['bbox'],
                        'vehicle_name': v['name']
                    })
        return detections, vehicles
class Store:
    def __init__(self, path):
        pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.lock = threading.RLock()
        with self.lock:
            self.db.executescript('''
                CREATE TABLE IF NOT EXISTS devices(stream TEXT PRIMARY KEY, name TEXT NOT NULL, role TEXT NOT NULL, anpr INTEGER NOT NULL DEFAULT 1, created_at INTEGER NOT NULL, url TEXT DEFAULT '', masks TEXT DEFAULT '[]', enabled INTEGER NOT NULL DEFAULT 1);
                CREATE TABLE IF NOT EXISTS sightings(id TEXT PRIMARY KEY, stream TEXT NOT NULL, plate TEXT NOT NULL, confidence REAL, ts INTEGER NOT NULL, photo BLOB, plate_photo BLOB);
                CREATE INDEX IF NOT EXISTS sight_plate ON sightings(plate, ts);
                CREATE INDEX IF NOT EXISTS sight_ts ON sightings(ts);
                CREATE TABLE IF NOT EXISTS passages(id INTEGER PRIMARY KEY AUTOINCREMENT, stream TEXT NOT NULL, count INTEGER, ts INTEGER NOT NULL);
                CREATE INDEX IF NOT EXISTS passage_ts ON passages(ts);
                CREATE TABLE IF NOT EXISTS settings(key TEXT PRIMARY KEY, value TEXT NOT NULL);
            ''')
            for col, col_def in [("url", "TEXT DEFAULT ''"), ("masks", "TEXT DEFAULT '[]'"), ("enabled", "INTEGER NOT NULL DEFAULT 1")]:
                try:
                    self.db.execute(f"ALTER TABLE devices ADD COLUMN {col} {col_def}")
                except Exception:
                    pass
            self.db.commit()
            try:
                states_sql = "','".join(s.upper() for s in INDIAN_STATES)
                self.db.execute("DELETE FROM sightings WHERE plate LIKE 'EX1T%' OR plate LIKE 'NO4L%' OR plate LIKE 'MEDAN%' OR plate LIKE 'GATE%' OR length(plate) < 8")
                self.db.execute(f"DELETE FROM sightings WHERE substr(plate, 1, 2) NOT IN ('{states_sql}')")
                self.db.commit()
            except Exception:
                pass
            try:
                cur_dev = self.db.cursor()
                cur_dev.execute("SELECT COUNT(*) FROM devices")
                if cur_dev.fetchone()[0] == 0:
                    for d in DEFAULT_VEHICLE_DEVICES:
                        self.add_device(d['stream'], d['name'], d['role'], d['anpr'], d['url'])
            except Exception as e:
                print("Default devices seed note:", e)
    def setting(self, key, fallback=None):
        with self.lock:
            row = self.db.execute('SELECT value FROM settings WHERE key=?', (key,)).fetchone()
        return row[0] if row else fallback
    def set_setting(self, key, value):
        with self.lock:
            self.db.execute('INSERT INTO settings VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value', (key, str(value)))
            self.db.commit()
    def rules(self):
        try:
            rules = json.loads(self.setting('rules', '{}'))
        except Exception:
            rules = {}
        return {'visit_gap_min': min(1440, max(1, int(rules.get('visit_gap_min', 30)))),
                'exit_grace_min': min(1440, max(1, int(rules.get('exit_grace_min', 45)))),
                'correlate_passages': bool(rules.get('correlate_passages', True)),
                'correlate_window_s': min(3600, max(30, int(rules.get('correlate_window_s', 300))))}
    def save_rules(self, body):
        rules = self.rules()
        for key in ('visit_gap_min', 'exit_grace_min', 'correlate_window_s'):
            if key in body:
                value = int(body[key])
                if not (1 <= value <= (3600 if key == 'correlate_window_s' else 1440)):
                    raise ValueError(f'{key} out of range')
                rules[key] = value
        if 'correlate_passages' in body:
            rules['correlate_passages'] = bool(body['correlate_passages'])
        self.set_setting('rules', json.dumps(rules))
        return self.rules()
    def devices(self):
        with self.lock:
            cols = [r[1] for r in self.db.execute('PRAGMA table_info(devices)').fetchall()]
            rows = self.db.execute(f"SELECT {','.join(cols)} FROM devices ORDER BY created_at").fetchall()
        result = []
        for r in rows:
            d = dict(zip(cols, r))
            d['anpr'] = bool(d.get('anpr', 1))
            if 'masks' in d:
                try:
                    d['masks'] = json.loads(d['masks']) if isinstance(d['masks'], str) else (d['masks'] or [])
                except Exception:
                    d['masks'] = []
            if 'enabled' in d:
                d['enabled'] = bool(d['enabled'])
            if 'url' not in d:
                d['url'] = ''
            result.append(d)
        return result
    def add_device(self, stream, name, role, anpr=True, url='', masks=None, enabled=True):
        url = str(url or '').strip()
        if not stream and url:
            clean = url.split('?')[0].rstrip('/')
            candidate = clean.split('/')[-1]
            stream = re.sub(r'[^A-Za-z0-9_-]', '', candidate)[:40]
            if not stream:
                stream = f'cam_{int(time.time())}'
        if not STREAM_RE.match(stream):
            raise ValueError('Invalid camera stream name')
        if role not in ROLES:
            raise ValueError('Role must be entry, exit or checkpoint')
        name = str(name or stream).strip()[:80] or stream
        masks_json = json.dumps(masks or [])
        with self.lock:
            cols = [r[1] for r in self.db.execute('PRAGMA table_info(devices)').fetchall()]
            data_map = {
                'stream': stream,
                'name': name,
                'role': role,
                'anpr': 1 if anpr else 0,
                'created_at': int(time.time() * 1000)
            }
            if 'masks' in cols:
                data_map['masks'] = masks_json
            if 'enabled' in cols:
                data_map['enabled'] = 1 if enabled else 0
            if 'url' in cols:
                data_map['url'] = url
            active_cols = [c for c in cols if c in data_map]
            placeholders = ','.join(['?'] * len(active_cols))
            col_names = ','.join(active_cols)
            update_clauses = ','.join([f"{c}=excluded.{c}" for c in active_cols if c != 'stream'])
            sql = f"INSERT INTO devices({col_names}) VALUES({placeholders}) ON CONFLICT(stream) DO UPDATE SET {update_clauses}"
            values = [data_map[c] for c in active_cols]
            self.db.execute(sql, values)
            self.db.commit()
        return [d for d in self.devices() if d['stream'] == stream][0]
    def remove_device(self, stream):
        with self.lock:
            if not self.db.execute('DELETE FROM devices WHERE stream=?', (stream,)).rowcount:
                raise ValueError('Unknown device')
            self.db.commit()
    def save_sighting(self, stream, plate, confidence, ts, photo=None, plate_photo=None):
        sid = uuid.uuid4().hex
        with self.lock:
            cols = [r[1] for r in self.db.execute('PRAGMA table_info(sightings)').fetchall()]
            data_map = {'id': sid, 'stream': stream, 'plate': plate, 'confidence': confidence, 'ts': ts, 'photo': photo, 'plate_photo': plate_photo, 'verified': 0}
            active_cols = [c for c in cols if c in data_map]
            placeholders = ','.join(['?'] * len(active_cols))
            self.db.execute(f"INSERT INTO sightings({','.join(active_cols)}) VALUES({placeholders})", [data_map[c] for c in active_cols])
            self.db.execute('DELETE FROM sightings WHERE ts<?', (ts - 30 * 86400000,))
            self.db.commit()
        return sid
    def sightings(self, since=0, plate=None, limit=500):
        query = 'SELECT id,stream,plate,confidence,ts,(photo IS NOT NULL),(plate_photo IS NOT NULL) FROM sightings WHERE ts>=?'
        params = [since]
        if plate:
            query += ' AND plate=?'
            params.append(plate.upper())
        query += ' ORDER BY ts LIMIT ?'
        params.append(limit)
        with self.lock:
            rows = self.db.execute(query, params).fetchall()
        return [{'id': r[0], 'stream': r[1], 'plate': r[2], 'confidence': r[3], 'ts': r[4],
                 'photo_url': f'/sightings/{r[0]}/photo' if r[5] else None,
                 'plate_url': f'/sightings/{r[0]}/plate' if r[6] else None} for r in rows]
    def sighting_photo(self, sid, kind='photo'):
        column = 'plate_photo' if kind == 'plate' else 'photo'
        with self.lock:
            row = self.db.execute(f'SELECT {column} FROM sightings WHERE id=?', (sid,)).fetchone()
        return row[0] if row and row[0] else None
    def save_passage(self, stream, count, ts):
        with self.lock:
            self.db.execute('INSERT INTO passages(stream,count,ts) VALUES(?,?,?)', (stream, count, ts))
            self.db.execute('DELETE FROM passages WHERE ts<?', (ts - 30 * 86400000,))
            self.db.commit()
    def passages(self, since=0):
        with self.lock:
            return [{'stream': r[0], 'count': r[1], 'ts': r[2]} for r in self.db.execute('SELECT stream,count,ts FROM passages WHERE ts>=? ORDER BY ts', (since,)).fetchall()]
def journeys(store, since_ms, plate_filter=None):
    """Cluster plate sightings into visits. A new visit starts after visit_gap_min
    of silence for the same plate. Exit is confirmed when the last sighting is on
    an 'exit' device, otherwise inferred after exit_grace_min of silence."""
    rules = store.rules()
    gap_ms = rules['visit_gap_min'] * 60000
    grace_ms = rules['exit_grace_min'] * 60000
    devices = {d['stream']: d for d in store.devices()}
    rows = store.sightings(since=since_ms, plate=plate_filter, limit=20000)
    by_plate = collections.defaultdict(list)
    for row in rows:
        by_plate[row['plate']].append(row)
    now_ms = int(time.time() * 1000)
    out = []
    for plate, reads in by_plate.items():
        visits, current = [], None
        for read in reads:
            if current is None or read['ts'] - current['reads'][-1]['ts'] > gap_ms:
                current = {'reads': []}
                visits.append(current)
            current['reads'].append(read)
        for visit in visits:
            reads_v = visit['reads']
            first, last = reads_v[0], reads_v[-1]
            cameras = []
            grouped = collections.OrderedDict()
            for read in reads_v:
                slot = grouped.setdefault(read['stream'], {'reads': 0, 'first': read['ts'], 'last': read['ts'], 'photo_url': None, 'confidence': 0})
                slot['reads'] += 1
                slot['last'] = read['ts']
                slot['confidence'] = max(slot['confidence'], read['confidence'] or 0)
                slot['photo_url'] = read['photo_url'] or slot['photo_url']
            for stream, slot in grouped.items():
                device = devices.get(stream, {})
                cameras.append({'stream': stream, 'name': device.get('name', stream), 'role': device.get('role', 'unassigned'), **slot})
            exit_device = devices.get(last['stream'], {})
            if exit_device.get('role') == 'exit':
                status, exited_at, exit_confirmed = 'exited', last['ts'], True
            elif now_ms - last['ts'] > grace_ms:
                status, exited_at, exit_confirmed = 'exited', last['ts'], False
            else:
                status, exited_at, exit_confirmed = 'on_site', None, False
            end = exited_at or now_ms
            out.append({'plate': plate, 'status': status, 'exit_confirmed': exit_confirmed,
                        'entry': {'stream': first['stream'], 'device': devices.get(first['stream'], {}).get('name', first['stream']), 'at': first['ts'], 'photo_url': first['photo_url'], 'plate_url': first['plate_url'], 'confidence': first['confidence']},
                        'exit': {'stream': last['stream'], 'device': devices.get(last['stream'], {}).get('name', last['stream']), 'at': exited_at, 'photo_url': last['photo_url'], 'confidence': last['confidence']} if exited_at else None,
                        'cameras': cameras, 'camera_count': len(cameras), 'reads': len(reads_v),
                        'first_at': first['ts'], 'last_at': last['ts'], 'dwell_ms': max(0, end - first['ts'])})
    out.sort(key=lambda v: v['first_at'], reverse=True)
    return out
def http_json(url, body=None, key=KEY, timeout=15):
    request = urllib.request.Request(url, data=json.dumps(body).encode() if body is not None else None,
                                     headers={'X-Local-Vision-Key': key, 'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())
class Engine:
    """Continuous multi-stream ANPR collector + objects passage poller."""
    def __init__(self, store):
        self.store = store
        self.detector = None
        self.model_lock = threading.Lock()
        self.lock = threading.Condition()
        self.workers = {}  # stream -> {latest, jpeg, sequence, seen, pending, noise, last_out, meta, gen, running}
        self.generation = 0
        self.obj_cursors = {}
        self.stats = {'scans': 0, 'sightings': 0, 'passages': 0, 'error': None}
        self.detector_error = None
        threading.Thread(target=self.infer_loop, daemon=True).start()
        threading.Thread(target=self.objects_loop, daemon=True).start()
    def ensure_detector(self):
        if self.detector is None and self.detector_error is None:
            try:
                self.detector = Detector()
            except Exception as error:
                self.detector_error = str(error)[:200]
        return self.detector
    def worker(self, stream):
        with self.lock:
            if stream not in self.workers:
                self.workers[stream] = {
                    'gen': self.generation,
                    'running': True,
                    'latest': None, 'jpeg': None, 'sequence': 0, 'seen': {}, 'pending': {}, 'noise': {},
                    'last_out': 0, 'meta': {'fps': 0, 'scan_ms': 0, 'plates': 0, 'error': None}
                }
                threading.Thread(target=self.capture, args=(stream, self.generation), daemon=True).start()
            return self.workers[stream]
    def drop(self, stream):
        with self.lock:
            w = self.workers.pop(stream, None)
            if w:
                w['running'] = False
            self.lock.notify_all()
    def sync(self):
        """Ensure a capture worker per ANPR-enabled device; stop removed ones."""
        wanted = {d['stream'] for d in self.store.devices() if d['anpr']}
        for stream in wanted:
            self.worker(stream)
        with self.lock:
            for stream in list(self.workers):
                if stream not in wanted:
                    w = self.workers.pop(stream, None)
                    if w:
                        w['running'] = False
            self.lock.notify_all()
    def capture(self, stream, my_gen):
        fps_history = collections.deque(maxlen=30)
        while True:
            with self.lock:
                w = self.workers.get(stream)
                if not w or w.get('gen') != my_gen or not w.get('running', True):
                    return
            dev = next((d for d in self.store.devices() if d['stream'] == stream), None)
            if CURRENT_FEED_MODE == 'demo':
                source = DEMO_VEHICLE_RECORDINGS.get(stream, str(RECORDINGS_DIR / "mgptz"))
            else:
                source = (dev.get('url') if dev and dev.get('url') else f'{RTSP_BASE}/{stream}')

            playlist = get_playlist(source)
            playlist_idx = 0
            current_source = playlist[playlist_idx] if playlist else source
            is_file = (playlist and len(playlist) > 0 and os.path.isfile(playlist[0])) or os.path.isfile(current_source)

            os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp|fflags;nobuffer|flags;low_delay'
            cap = cv2.VideoCapture(current_source, cv2.CAP_FFMPEG, [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000, cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000])
            if not cap.isOpened():
                with self.lock:
                    if stream in self.workers and self.workers[stream].get('gen') == my_gen:
                        self.workers[stream]['meta']['fps'] = 0
                        self.workers[stream]['meta']['error'] = 'Camera connection unavailable; auto-retry in 15s'
                cap.release()
                time.sleep(15)
                continue
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass

            consecutive_fails = 0
            video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            frame_delay = 1.0 / max(10.0, min(30.0, video_fps)) if is_file else 0.0

            while True:
                t_frame_start = time.monotonic()
                with self.lock:
                    w = self.workers.get(stream)
                    if not w or w.get('gen') != my_gen or not w.get('running', True):
                        cap.release()
                        return

                ok, frame = cap.read()
                if not ok:
                    if is_file and playlist:
                        playlist_idx = (playlist_idx + 1) % len(playlist)
                        current_source = playlist[playlist_idx]
                        cap.release()
                        cap = cv2.VideoCapture(current_source, cv2.CAP_FFMPEG)
                        time.sleep(0.02)
                        continue
                    consecutive_fails += 1
                    if consecutive_fails > 10:
                        with self.lock:
                            if stream in self.workers and self.workers[stream].get('gen') == my_gen:
                                self.workers[stream]['meta']['fps'] = 0
                                self.workers[stream]['meta']['error'] = 'Camera connection unavailable; auto-retry in 15s'
                        cap.release()
                        time.sleep(15)
                        break
                    time.sleep(0.05)
                    continue

                consecutive_fails = 0
                now = time.monotonic()
                with self.lock:
                    w = self.workers.get(stream)
                    if not w or w.get('gen') != my_gen or not w.get('running', True):
                        cap.release()
                        return
                    w['latest'] = (frame, now)
                    w['meta']['error'] = None
                    active_v, active_d = w.get('active_boxes', ([], []))

                hud_frame = frame.copy()
                for v in active_v:
                    vx1, vy1, vx2, vy2 = v['bbox']
                    cv2.rectangle(hud_frame, (vx1, vy1), (vx2, vy2), (240, 160, 40), 2)
                    cv2.putText(hud_frame, f"{v['name']} {int(v['score']*100)}%", (vx1, max(18, vy1 - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 160, 40), 2, cv2.LINE_AA)

                for det in active_d:
                    x, y, w, h = det['bbox']
                    cv2.polylines(hud_frame, [det['quad'].astype(np.int32)], True, (60, 220, 120), 2)
                    cv2.putText(hud_frame, det['plate'], (x, max(15, y - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 220, 120), 2, cv2.LINE_AA)

                # Resize to 768x432 for optimal web streaming throughput
                if hud_frame.shape[1] > 768:
                    stream_frame = cv2.resize(hud_frame, (768, int(hud_frame.shape[0] * 768 / hud_frame.shape[1])), interpolation=cv2.INTER_LINEAR)
                else:
                    stream_frame = hud_frame

                ok_enc, jpeg = cv2.imencode('.jpg', stream_frame, [cv2.IMWRITE_JPEG_QUALITY, 58])
                if ok_enc:
                    fps_history.append(now)
                    cutoff = now - 1.0
                    while fps_history and fps_history[0] < cutoff:
                        fps_history.popleft()
                    measured_fps = round(len(fps_history) / max(now - fps_history[0], 0.001), 1) if len(fps_history) > 1 else (video_fps if is_file else 25.0)
                    display_fps = min(max(measured_fps, 10.0), 30.0)

                    with self.lock:
                        w = self.workers.get(stream)
                        if w and w.get('gen') == my_gen and w.get('running', True):
                            w['jpeg'] = jpeg.tobytes()
                            w['sequence'] += 1
                            w['meta']['fps'] = display_fps
                            self.lock.notify_all()

                # Accurate real-time frame pacing (1.0x normal speed playback)
                if is_file:
                    elapsed = time.monotonic() - t_frame_start
                    sleep_needed = frame_delay - elapsed
                    if sleep_needed > 0:
                        time.sleep(sleep_needed)

            cap.release()
            time.sleep(1.0)
    def canonical(self, worker, plate, score=0.85):
        for other, item in worker['pending'].items():
            if len(other) == len(plate) and sum(a != b for a, b in zip(other, plate)) <= 2:
                cands = item.setdefault('candidates', {})
                cands[plate] = cands.get(plate, 0.0) + score
                consensus = max(cands.items(), key=lambda x: x[1])[0]
                return consensus
        return plate
    def infer_loop(self):
        last_frame = {}
        last_infer_time = {}
        while True:
            now = time.monotonic()
            with self.lock:
                items = [(stream, w['latest'], w) for stream, w in self.workers.items()]
            # Temporal Decimation: pace inference to ~5 FPS (200ms interval) per camera
            items = [(s, latest, w) for s, latest, w in items
                     if latest and last_frame.get(s) != latest[1] and (now - last_infer_time.get(s, 0)) >= 0.20]
            if not items:
                time.sleep(0.03)
                continue
            for stream, (original, captured), worker in items:
                last_frame[stream] = captured
                last_infer_time[stream] = time.monotonic()
                detector = self.ensure_detector()
                if detector is None:
                    time.sleep(1)
                    break
                frame = original.copy()
                try:
                    with self.model_lock:
                        scan_start = time.monotonic()
                        detections, vehicles = detector.scan(frame)
                        scan_ms = (time.monotonic() - scan_start) * 1000

                    # Draw confirmed vehicle bounding boxes
                    for v in vehicles:
                        vx1, vy1, vx2, vy2 = v['bbox']
                        cv2.rectangle(frame, (vx1, vy1), (vx2, vy2), (240, 160, 40), 2)
                        cv2.putText(frame, f"{v['name']} {int(v['score']*100)}%", (vx1, max(18, vy1 - 6)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 160, 40), 2, cv2.LINE_AA)

                    now = time.monotonic()
                    now_ms = int(time.time() * 1000)
                    with self.lock:
                        worker['seen'] = {p: ts for p, ts in worker['seen'].items() if now - ts < 45}
                        worker['pending'] = {p: i for p, i in worker['pending'].items() if now - i['last'] < 20}
                        for det in detections:
                            det['plate'] = self.canonical(worker, det['plate'], det['score'])
                            x, y, w, h = det['bbox']
                            cv2.polylines(frame, [det['quad'].astype(np.int32)], True, (60, 220, 120), 2)
                            cv2.putText(frame, det['plate'], (x, max(15, y - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 220, 120), 2, cv2.LINE_AA)
                            if det['plate'] in worker['seen']:
                                continue
                            center = det['quad'].mean(axis=0)
                            previous = worker['pending'].get(det['plate'])
                            sightings = (previous['count'] if previous else 0) + 1
                            cands = (previous.get('candidates') if previous else {}) or {det['plate']: det['score']}
                            worker['pending'][det['plate']] = {'count': sightings, 'center': center, 'last': now, 'candidates': cands}
                            if sightings < 2 and det['score'] < 0.80:
                                continue
                            worker['seen'][det['plate']] = now

                            # Save full vehicle crop for sighting photo
                            vbox = det.get('vehicle_bbox')
                            if vbox:
                                vx1, vy1, vx2, vy2 = vbox
                                crop = frame[max(0, vy1 - 15):min(frame.shape[0], vy2 + 15), max(0, vx1 - 15):min(frame.shape[1], vx2 + 15)]
                            else:
                                margin = int(max(w, h) * 1.4)
                                crop = frame[max(0, y - margin):min(frame.shape[0], y + h + margin), max(0, x - margin):min(frame.shape[1], x + w + margin)]

                            ok, photo = cv2.imencode('.jpg', crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
                            ok2, plate_img = cv2.imencode('.jpg', frame[max(0, y - 8):min(frame.shape[0], y + h + 8), max(0, x - 8):min(frame.shape[1], x + w + 8)], [cv2.IMWRITE_JPEG_QUALITY, 90])
                            if ok:
                                self.store.save_sighting(stream, det['plate'], det['score'], now_ms,
                                                         photo.tobytes(), plate_img.tobytes() if ok2 else None)
                                self.stats['sightings'] += 1
                        worker['meta'].update(scan_ms=round(scan_ms, 1), plates=len(detections),
                                              frame_age_ms=round((time.monotonic() - captured) * 1000))
                        self.stats['scans'] += 1
                    with self.lock:
                        if stream in self.workers:
                            worker['active_boxes'] = (vehicles, detections)
                except Exception:
                    with self.lock:
                        worker['meta']['error'] = 'ANPR inference error'
                    time.sleep(0.5)
    def objects_loop(self):
        while True:
            try:
                for base in OBJECTS_URLS:
                    cursor = self.obj_cursors.get(base, 0)
                    result = http_json(f'{base}/events?after={cursor}', timeout=10)
                    for event in result.get('events', []):
                        self.obj_cursors[base] = max(cursor, event.get('id', 0))
                        counts = event.get('counts') or {}
                        if counts.get('vehicle', 0) > 0:
                            self.store.save_passage(event.get('stream', 'unknown'), counts['vehicle'], int(event.get('ts') or time.time() * 1000))
                            self.stats['passages'] += 1
                self.stats['error'] = None
            except Exception as error:
                self.stats['error'] = str(error)[:160]
            time.sleep(2)
    def state(self):
        with self.lock:
            workers = {s: {'jpeg': w['jpeg'] is not None, **w['meta']} for s, w in self.workers.items()}
        return {'running': True, 'model': 'PP-OCRv3 + CRNN' if self.detector else 'loading',
                'model_error': self.detector_error, 'workers': workers, 'objects_backends': len(OBJECTS_URLS), **self.stats}

    def get_feed_mode(self):
        global CURRENT_FEED_MODE
        return CURRENT_FEED_MODE

    def set_feed_mode(self, mode=None):
        global CURRENT_FEED_MODE, DB_PATH
        if not mode:
            mode = 'demo' if CURRENT_FEED_MODE == 'live' else 'live'
        mode = 'demo' if str(mode).lower() == 'demo' else 'live'
        
        if mode == CURRENT_FEED_MODE:
            return {'ok': True, 'mode': mode, 'active_db': os.path.basename(DB_PATH)}
            
        print(f"[VEHICLE TRACKER] Switching feed mode: {CURRENT_FEED_MODE} -> {mode}")
        CURRENT_FEED_MODE = mode
        
        try:
            with open(MODE_STATE_FILE, 'w') as f:
                json.dump({'mode': mode}, f)
        except Exception:
            pass
            
        if mode == 'demo':
            DB_PATH = DEMO_DB_PATH
            new_store = Store(DEMO_DB_PATH)
            if len(new_store.devices()) == 0:
                for d in self.store.devices():
                    new_store.add_device(d['stream'], d['name'], d['role'], d.get('anpr', 1), d.get('url', ''))
            self.store = new_store
            Handler.store = new_store
        else:
            DB_PATH = LIVE_DB_PATH
            new_store = Store(LIVE_DB_PATH)
            self.store = new_store
            Handler.store = new_store

        with self.lock:
            self.generation += 1
            for s, w in self.workers.items():
                w['running'] = False
            self.workers.clear()
            self.lock.notify_all()

        time.sleep(0.15)
        self.sync()
        return {'ok': True, 'mode': mode, 'active_db': os.path.basename(DB_PATH)}
class Handler(BaseHTTPRequestHandler):
    store = None
    engine = None
    def log_message(self, *args):
        pass
    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS, PUT, DELETE, PATCH')
        self.send_header('Access-Control-Allow-Headers', '*')
        self.end_headers()
    def send_json(self, status, data):
        raw = json.dumps(data).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(raw)))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(raw)
    def authorized(self):
        if not KEY:
            return True
        header_key = self.headers.get('X-Local-Vision-Key', '')
        if header_key and hmac.compare_digest(header_key, KEY):
            return True
        # Allow requests from browser / same network / localhost
        origin = self.headers.get('Origin', '')
        referer = self.headers.get('Referer', '')
        sec_fetch = self.headers.get('Sec-Fetch-Mode', '')
        if sec_fetch in ('cors', 'navigate', 'no-cors') or origin or referer or self.client_address[0] in ('127.0.0.1', '::1', '172.17.0.1'):
            return True
        return False
    def body(self):
        size = int(self.headers.get('Content-Length', '0'))
        if size > 1024 * 1024:
            raise ValueError('Request too large')
        return json.loads(self.rfile.read(size) or b'{}')
    def do_GET(self):
        parsed = urlparse(self.path)
        path_str = parsed.path
        if path_str.startswith('/api'):
            path_str = path_str[4:] or '/'
        query = parse_qs(parsed.query)
        try:
            if path_str == '/health':
                return self.send_json(200, {'ok': True})
            if path_str in ('/feed-source', '/api/feed-source'):
                return self.send_json(200, {
                    'mode': self.engine.get_feed_mode(),
                    'active_db': os.path.basename(DB_PATH)
                })
            if path_str == '/state':
                return self.send_json(200, {
                    'devices': self.store.devices(),
                    'rules': self.store.rules(),
                    'engine': self.engine.state(),
                    'feed_mode': self.engine.get_feed_mode()
                })
            if path_str == '/devices':
                return self.send_json(200, {'devices': self.store.devices()})
            if path_str == '/rules':
                return self.send_json(200, self.store.rules())
            if path_str == '/journeys':
                hours = min(720, max(1, float(query.get('hours', ['24'])[0])))
                plate = query.get('plate', [None])[0]
                result = journeys(self.store, int(time.time() * 1000 - hours * 3600000), plate)
                return self.send_json(200, {'journeys': result, 'count': len(result)})
            if path_str in ('/journeys.csv', '/journeys/export'):
                hours = min(720, max(1, float(query.get('hours', ['24'])[0])))
                plate = query.get('plate', [None])[0]
                status_filter = query.get('status', [None])[0]
                camera_filter = query.get('camera', [None])[0]
                result = journeys(self.store, int(time.time() * 1000 - hours * 3600000), plate)
                if status_filter:
                    result = [v for v in result if v['status'] == status_filter]
                if camera_filter:
                    result = [v for v in result if any(c['stream'] == camera_filter for c in v['cameras'])
                              or (v.get('entry') or {}).get('stream') == camera_filter
                              or (v.get('exit') or {}).get('stream') == camera_filter]
                import csv, io
                out = io.StringIO()
                writer = csv.writer(out)
                writer.writerow(['S.No', 'License Plate', 'Status', 'Entry Camera', 'Entry Date & Time',
                                 'Exit Camera', 'Exit Date & Time', 'Dwell Duration', 'Dwell (Minutes)',
                                 'Cameras Visited', 'Total Reads', 'Entry Vehicle Photo URL', 'Exit Vehicle Photo URL'])
                host_header = self.headers.get('Host', '124.123.89.162:31016')
                base_url = f"http://{host_header}" if not host_header.startswith('http') else host_header
                for idx, j in enumerate(result, 1):
                    first_ts = to_ist_strftime(j['first_at'] / 1000) if j.get('first_at') else '—'
                    last_ts = to_ist_strftime(j['exit']['at'] / 1000) if j.get('exit') and j['exit'].get('at') else '—'
                    dwell_sec = int(j.get('dwell_ms', 0) / 1000)
                    dwell_str = f"{dwell_sec // 60}m {dwell_sec % 60}s" if dwell_sec >= 60 else f"{dwell_sec}s"
                    dwell_min = round(dwell_sec / 60.0, 1)
                    cams = ' -> '.join(c['name'] for c in j.get('cameras', [])) or j.get('entry', {}).get('device', '—')
                    entry_info = j.get('entry') or {}
                    exit_info = j.get('exit') or {}
                    entry_photo = f"{base_url}{entry_info['photo_url']}" if entry_info.get('photo_url') else ''
                    exit_photo = f"{base_url}{exit_info['photo_url']}" if exit_info.get('photo_url') else ''
                    writer.writerow([idx, j['plate'], j['status'].upper(), entry_info.get('device', '—'),
                                     first_ts, exit_info.get('device', '—'),
                                     last_ts, dwell_str, dwell_min, cams, j.get('reads', 1), entry_photo, exit_photo])
                payload = out.getvalue().encode('utf-8-sig')
                self.send_response(200)
                self.send_header('Content-Type', 'text/csv; charset=utf-8')
                self.send_header('Content-Disposition', 'attachment; filename="Medanta_Vehicle_Journeys_' + to_ist_strftime(time.time(), '%Y%m%d_%H%M%S') + '.csv"')
                self.send_header('Content-Length', str(len(payload)))
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                return self.wfile.write(payload)
            if path_str == '/sightings':
                hours = min(720, max(1, float(query.get('hours', ['24'])[0])))
                plate = query.get('plate', [None])[0]
                limit = min(2000, max(1, int(query.get('limit', ['500'])[0])))
                return self.send_json(200, {'sightings': self.store.sightings(int(time.time() * 1000 - hours * 3600000), plate, limit)})
            if path_str == '/passages':
                hours = min(720, max(1, float(query.get('hours', ['24'])[0])))
                return self.send_json(200, {'passages': self.store.passages(int(time.time() * 1000 - hours * 3600000))})
            if path_str == '/analytics':
                hours = min(720, max(1, float(query.get('hours', ['24'])[0])))
                since = int(time.time() * 1000 - hours * 3600000)
                visits = journeys(self.store, since)
                # Compute today's start in IST timezone
                now_sec = int(time.time())
                seconds_today_in_ist = (now_sec + IST_OFFSET_SEC) % 86400
                today = (now_sec - seconds_today_in_ist) * 1000
                dwell = [v['dwell_ms'] for v in visits if v['status'] == 'exited']
                per_camera = collections.Counter()
                for v in visits:
                    for cam in v['cameras']:
                        per_camera[cam['stream']] += cam['reads']
                bucket_ms = 3600000
                hourly = collections.Counter()
                for v in visits:
                    # Align to IST hourly buckets
                    ist_bucket = (v['first_at'] + IST_OFFSET_MS) // bucket_ms * bucket_ms - IST_OFFSET_MS
                    hourly[ist_bucket] += 1
                return self.send_json(200, {
                    'vehicles_today': sum(1 for v in visits if v['first_at'] >= today),
                    'on_site': sum(1 for v in visits if v['status'] == 'on_site'),
                    'exited': sum(1 for v in visits if v['status'] == 'exited'),
                    'unique_plates': len({v['plate'] for v in visits}),
                    'avg_dwell_ms': int(sum(dwell) / len(dwell)) if dwell else None,
                    'per_camera': dict(per_camera),
                    'hourly_entries': [{'bucket': ts, 'count': n} for ts, n in sorted(hourly.items())],
                })
            match = re.match(r'^/sightings/([a-f0-9]+)/(photo|plate)$', path_str)
            if match:
                blob = self.store.sighting_photo(match.group(1), match.group(2))
                if not blob:
                    return self.send_json(404, {'error': 'No image'})
                self.send_response(200)
                self.send_header('Content-Type', 'image/jpeg')
                self.send_header('Content-Length', str(len(blob)))
                self.send_header('Access-Control-Allow-Origin', '*')
                self.send_header('Cache-Control', 'private, max-age=3600')
                self.end_headers()
                return self.wfile.write(blob)
            match = re.match(r'^/devices/([A-Za-z0-9_-]+)/frame.jpg$', path_str)
            if match:
                with self.engine.lock:
                    worker = self.engine.workers.get(match.group(1))
                    blob = worker['jpeg'] if worker else None
                if not blob:
                    return self.send_json(404, {'error': 'No frame yet'})
                self.send_response(200)
                self.send_header('Content-Type', 'image/jpeg')
                self.send_header('Content-Length', str(len(blob)))
                self.send_header('Access-Control-Allow-Origin', '*')
                self.send_header('Cache-Control', 'no-store')
                self.end_headers()
                return self.wfile.write(blob)
            match = re.match(r'^/devices/([A-Za-z0-9_-]+)/stream.mjpg$', path_str)
            if match:
                stream = match.group(1)
                req_mode = query.get('mode', [None])[0]
                if req_mode and req_mode != self.engine.get_feed_mode():
                    return self.send_json(409, {'error': f'Mode mismatch: requested {req_mode}, active is {self.engine.get_feed_mode()}'})

                with self.engine.lock:
                    worker = self.engine.workers.get(stream)
                if not worker:
                    return self.send_json(404, {'error': 'No ANPR worker on this device'})
                try:
                    self.request.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    self.request.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 65536)
                except Exception:
                    pass
                self.send_response(200)
                self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
                self.send_header('Pragma', 'no-cache')
                self.end_headers()
                sequence = -1
                stream_gen = worker.get('gen', 0)
                try:
                    while True:
                        if req_mode and req_mode != self.engine.get_feed_mode():
                            return
                        frame = None
                        with self.engine.lock:
                            while stream in self.engine.workers and (self.engine.workers[stream]['jpeg'] is None or self.engine.workers[stream]['sequence'] == sequence):
                                if not self.engine.lock.wait(timeout=0.5):
                                    break
                            if stream not in self.engine.workers:
                                return
                            cur = self.engine.workers[stream]
                            if cur.get('gen', 0) != stream_gen:
                                return
                            if cur['jpeg'] is not None and cur['sequence'] != sequence:
                                sequence = cur['sequence']
                                frame = cur['jpeg']
                        if frame is not None:
                            self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: ' + str(len(frame)).encode() + b'\r\n\r\n' + frame + b'\r\n')
                            self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError, socket.error):
                    return
            # Serve static UI files from DIST_DIR
            if DIST_DIR.exists():
                rel = path_str.lstrip('/')
                target = (DIST_DIR / rel).resolve()
                if not target.is_relative_to(DIST_DIR.resolve()) or not target.exists() or target.is_dir():
                    target = DIST_DIR / 'index.html'
                if target.is_file():
                    ext = target.suffix.lower()
                    mime = MIME_TYPES.get(ext, 'application/octet-stream')
                    content = target.read_bytes()
                    self.send_response(200)
                    self.send_header('Content-Type', mime)
                    self.send_header('Content-Length', str(len(content)))
                    self.send_header('Access-Control-Allow-Origin', '*')
                    if ext in ('.js', '.css', '.woff2'):
                        self.send_header('Cache-Control', 'public, max-age=86400')
                    else:
                        self.send_header('Cache-Control', 'no-cache')
                    self.end_headers()
                    return self.wfile.write(content)
            return self.send_json(404, {'error': 'Not found'})
        except Exception as error:
            return self.send_json(400, {'error': str(error)[:200]})
    def do_POST(self):
        if not self.authorized():
            return self.send_json(401, {'error': 'Authorization required'})
        parsed = urlparse(self.path)
        path_str = parsed.path
        if path_str.startswith('/api'):
            path_str = path_str[4:] or '/'
        try:
            data = self.body()
            if path_str in ('/feed-source', '/api/feed-source', '/feed-source/toggle', '/api/feed-source/toggle'):
                mode = data.get('mode')
                res = self.engine.set_feed_mode(mode)
                return self.send_json(200, res)
            if path_str == '/devices':
                device = self.store.add_device(
                    stream=data.get('stream', ''),
                    name=data.get('name'),
                    role=data.get('role', 'checkpoint'),
                    anpr=data.get('anpr', True),
                    url=data.get('url', ''),
                    masks=data.get('masks'),
                    enabled=data.get('enabled', True)
                )
                self.engine.drop(device['stream'])
                self.engine.sync()
                return self.send_json(200, {'device': device, 'devices': self.store.devices()})
            if path_str == '/devices/toggle':
                devices = self.store.set_device_enabled(data.get('stream', ''), bool(data.get('enabled', True)))
                self.engine.sync()
                return self.send_json(200, {'devices': devices})
            if path_str == '/devices/delete':
                self.store.remove_device(data.get('stream', ''))
                self.engine.sync()
                return self.send_json(200, {'devices': self.store.devices()})
            if path_str == '/rules':
                return self.send_json(200, self.store.save_rules(data))
            if path_str == '/sightings':  # manual import (e.g. from anpr-gpu events replay)
                stream, plate = data.get('stream', ''), str(data.get('plate', '')).upper()
                if not STREAM_RE.match(stream) or not re.match(r'^[A-Z0-9]{4,12}$', plate):
                    raise ValueError('Valid stream and plate required')
                photo = base64.b64decode(data['photo_b64']) if data.get('photo_b64') else None
                plate_photo = base64.b64decode(data['plate_b64']) if data.get('plate_b64') else None
                sid = self.store.save_sighting(stream, plate, data.get('confidence'), int(data.get('ts') or time.time() * 1000), photo, plate_photo)
                return self.send_json(200, {'id': sid})
            return self.send_json(404, {'error': 'Not found'})
        except Exception as error:
            return self.send_json(400, {'error': str(error)[:200]})
def main():
    store = Store(DB_PATH)
    engine = Engine(store)
    Handler.store = store
    Handler.engine = engine
    engine.sync()
    bind = os.environ.get('TRACKER_BIND', '0.0.0.0')
    port = int(os.environ.get('TRACKER_PORT', '8176'))
    print(f'Vehicle tracker ready on {bind}:{port}', flush=True)
    ThreadingHTTPServer((bind, port), Handler).serve_forever()
if __name__ == '__main__':
    main()