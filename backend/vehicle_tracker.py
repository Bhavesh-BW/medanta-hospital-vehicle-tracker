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
import html
import json
import os
import pathlib
import re
import socket
import sqlite3
import threading
import time
import urllib.error
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

if LPD_YuNet is not None:
    class MultiPlateYuNet(LPD_YuNet):
        """YuNet subclass ensuring axis-aligned NMS boxes and cached prior generation."""
        def setInputSize(self, inputSize):
            if tuple(self.input_size) == tuple(inputSize):
                return
            self.input_size = np.array(inputSize)
            self._priorGen()

        def _postprocess(self, blob):
            dets = self._decode(blob)
            pts = dets[:, :8].reshape(-1, 4, 2)
            x_min = pts[:, :, 0].min(axis=1)
            y_min = pts[:, :, 1].min(axis=1)
            bw = pts[:, :, 0].max(axis=1) - x_min
            bh = pts[:, :, 1].max(axis=1) - y_min
            bboxes = np.stack([x_min, y_min, bw, bh], axis=1).tolist()
            keepIdx = cv2.dnn.NMSBoxes(
                bboxes=bboxes,
                scores=dets[:, -1].tolist(),
                score_threshold=self.confidence_threshold,
                nms_threshold=self.nms_threshold,
                top_k=self.top_k
            )
            if len(keepIdx) > 0:
                keepIdx = np.array(keepIdx).reshape(-1)
                return dets[keepIdx][:self.keep_top_k]
            return np.empty(shape=(0, 9))
else:
    MultiPlateYuNet = None
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
EMAIL_DISPATCH_URLS = [
    u.strip() for u in os.environ.get(
        'EMAIL_DISPATCH_URL',
        'http://172.17.0.1:31040/api/email/dispatch,http://127.0.0.1:31040/api/email/dispatch,http://124.123.89.162:31040/api/email/dispatch'
    ).split(',') if u.strip()
]

# IST Timezone helper (+05:30)
IST_OFFSET_SEC = int(5.5 * 3600)
IST_OFFSET_MS = int(5.5 * 3600 * 1000)

def to_ist_strftime(epoch_sec=None, fmt='%Y-%m-%d %H:%M:%S'):
    sec = epoch_sec if epoch_sec is not None else time.time()
    return time.strftime(fmt, time.gmtime(sec + IST_OFFSET_SEC))
STREAM_RE = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_-]{0,40}$')
ROLES = ('entry', 'exit', 'basement_entry', 'basement_exit', 'basement', 'porch', 'emergency', 'checkpoint')
GATE_DIRECTIONS = ('incoming', 'outgoing', 'basement_incoming', 'basement_outgoing')
COUNTABLE_ROLE_DIRECTION = {
    'entry': 'incoming',
    'exit': 'outgoing',
    'basement_entry': 'basement_incoming',
    'basement': 'basement_incoming',
    'basement_exit': 'basement_outgoing',
}
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
    # Top 8% (timestamp bar: e.g. "11-12-2026 Thu 15:23:44")
    if cy < int(h * 0.08):
        return True
    # Very bottom edge strip (5%)
    if cy > int(h * 0.95):
        return True
    # Bottom-right corner label (where "Exit gate", "Main Gate Outside" is stamped)
    if cx > int(w * 0.75) and cy > int(h * 0.88):
        return True
    # Bottom-left corner label
    if cx < int(w * 0.22) and cy > int(h * 0.90):
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
    # Strip single leading edge/IND-hologram noise character (e.g. 'EHR55BF1916' -> 'HR55BF1916')
    if len(t) >= 9 and t[:2] not in INDIAN_STATES and t[1:3] in INDIAN_STATES:
        tail = t[-4:]
        if sum(ch.isdigit() or ch in TO_DIGIT for ch in tail) == 4:
            t = t[1:]
    return t
_STATE_LETTER_CONFUSABLES = {
    'H': set('MNKB'), 'D': set('OQ'), 'L': set('IU'), 'R': set('BPK'),
    'U': set('VO'), 'P': set('RFB'), 'B': set('RPD'), 'M': set('NHW'),
    'N': set('MH'), 'C': set('GO'), 'G': set('C'), 'J': set('I'),
    'K': set('XH'), 'A': set('H'), 'W': set('MV'),
}

def fix_plate(raw_text, ocr_conf=None):
    if is_garbage_text(raw_text):
        return None
    text = clean_indian_text(raw_text)
    if is_garbage_text(text) or len(text) < 7 or len(text) > 11:
        return None

    fixes = 0
    # First 2 chars must be State letters
    s0 = TO_LETTER.get(text[0], text[0])
    s1 = TO_LETTER.get(text[1], text[1])
    if s0 != text[0]:
        fixes += 1
    if s1 != text[1]:
        fixes += 1
    if not (s0.isalpha() and s1.isalpha()):
        return None
    state = s0 + s1

    # Last 4 characters must be digits
    tail_chars = text[-4:]
    fixed_tail = []
    for ch in tail_chars:
        if ch.isdigit():
            fixed_tail.append(ch)
        elif ch in TO_DIGIT:
            fixed_tail.append(TO_DIGIT[ch])
            fixes += 1
        else:
            return None
    fixed_tail_str = ''.join(fixed_tail)

    if state not in INDIAN_STATES:
        # Strict single-char priority state recovery only on clear full plates
        if len(text) < 9 or (ocr_conf is not None and ocr_conf < 0.65) or fixes > 1:
            return None
        best_state, best_rank = None, -1
        for idx_p, s in enumerate(PRIORITY_STATES):
            match0 = (s[0] == s0)
            match1 = (s[1] == s1)
            if not (match0 or match1):
                continue
            rank = 2 if (
                (match0 and s1 in _STATE_LETTER_CONFUSABLES.get(s[1], ()))
                or (match1 and s0 in _STATE_LETTER_CONFUSABLES.get(s[0], ()))
            ) else 1
            if rank > best_rank:
                best_rank, best_state = rank, s
        if not best_state or (best_rank < 2 and ocr_conf is not None and ocr_conf < 0.75):
            return None
        state = best_state
        fixes += 1

    mid = text[2:-4]
    if not mid:
        return None

    # Determine RTO digits and Series letters
    rto = []
    if state == 'DL':
        if mid[0].isdigit():
            rto.append(mid[0])
        elif mid[0] in TO_DIGIT:
            rto.append(TO_DIGIT[mid[0]])
            fixes += 1
        else:
            return None
        start_idx = 1
        if len(mid) >= 2 and (mid[1].isdigit() or (mid[1] in TO_DIGIT and len(mid) >= 3 and mid[2].isalpha())):
            if mid[1].isdigit():
                rto.append(mid[1])
            else:
                rto.append(TO_DIGIT[mid[1]])
                fixes += 1
            start_idx = 2
    else:
        if mid[0].isdigit():
            rto.append(mid[0])
        elif mid[0] in TO_DIGIT:
            rto.append(TO_DIGIT[mid[0]])
            fixes += 1
        else:
            return None
        start_idx = 1
        if len(mid) >= 2 and (mid[1].isdigit() or (mid[1] in TO_DIGIT and len(mid) >= 3)):
            if mid[1].isdigit():
                rto.append(mid[1])
            else:
                rto.append(TO_DIGIT[mid[1]])
                fixes += 1
            start_idx = 2

    series = []
    for ch in mid[start_idx:]:
        if ch.isalpha():
            series.append(ch)
        elif ch in TO_LETTER:
            series.append(TO_LETTER[ch])
            fixes += 1
        else:
            return None
    if not series or len(series) > 3 or fixes > 4:
        return None

    candidate = state + ''.join(rto) + ''.join(series) + fixed_tail_str
    if 8 <= len(candidate) <= 10 and STRICT_RE.match(candidate.lower()):
        return candidate
    return None
cv2.setNumThreads(4)
VEHICLE_TYPES = ('car', 'bike', 'auto', 'ambulance')
_VC_CONFUSABLE = str.maketrans({'O': '0', 'Q': '0', 'I': '1', 'L': '1', 'Z': '2', 'S': '5', 'G': '6', 'B': '8'})

def normalize_vehicle_type(name):
    """Map model/legacy vehicle class names onto the 4 tracked security categories."""
    t = str(name or 'car').strip().lower()
    if t == 'ambulance':
        return 'ambulance'
    if t in ('auto', 'rickshaw', 'autorickshaw', 'three_wheeler'):
        return 'auto'
    if t in ('bike', 'motorcycle', 'motorbike', 'scooter', 'two_wheeler'):
        return 'bike'
    return 'car'

def normalize_gate_group(group, display=False):
    """Normalize optional gate/location group name so cameras monitoring one physical gate dedup cleanly."""
    clean = re.sub(r'\s+', ' ', str(group or '').strip())[:60]
    return clean if display else clean.lower()

def _levenshtein_le_one(a, b):
    """Return True if Levenshtein edit distance between strings a and b is <= 1."""
    if a == b:
        return True
    la, lb = len(a), len(b)
    if abs(la - lb) > 1:
        return False
    if la == lb:
        return sum(1 for x, y in zip(a, b) if x != y) <= 1
    shorter, longer = (a, b) if la < lb else (b, a)
    i = j = skips = 0
    while i < len(shorter) and j < len(longer):
        if shorter[i] == longer[j]:
            i += 1
            j += 1
        else:
            skips += 1
            if skips > 1:
                return False
            j += 1
    return True

def are_plates_similar_vc(p1, p2, allow_single_edit=False):
    """Compare plates tolerating OCR confusables and optional 1-char edit distance."""
    if not p1 or not p2:
        return False
    a = re.sub(r'[^A-Z0-9]', '', str(p1).upper())
    b = re.sub(r'[^A-Z0-9]', '', str(p2).upper())
    if not a or not b:
        return False
    if a == b:
        return True
    na, nb = a.translate(_VC_CONFUSABLE), b.translate(_VC_CONFUSABLE)
    if na == nb:
        return True
    if not allow_single_edit or min(len(na), len(nb)) < 7:
        return False
    if len(na) == len(nb):
        return sum(1 for x, y in zip(na, nb) if x != y) <= 1
    if abs(len(na) - len(nb)) == 1 and (na[-3:] == nb[-3:] or na[:4] == nb[:4]):
        return _levenshtein_le_one(na, nb)
    return False

def _merge_stream_csv(raw_csv, *extra_streams):
    """Build an ordered unique comma-separated camera stream list."""
    ordered = []
    seen = set()
    for token in str(raw_csv or '').split(',') + list(extra_streams):
        s = str(token or '').strip()
        if s and s not in seen:
            seen.add(s)
            ordered.append(s)
    return ','.join(ordered)

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

    @staticmethod
    def _letterbox(frame, new_shape=(640, 640), color=(114, 114, 114)):
        """Resize keeping aspect ratio and pad to square (matches Ultralytics YOLO training preprocessing)."""
        h, w = frame.shape[:2]
        r = min(new_shape[0] / max(1, h), new_shape[1] / max(1, w))
        new_unpad = (max(1, int(round(w * r))), max(1, int(round(h * r))))
        dw = (new_shape[1] - new_unpad[0]) / 2.0
        dh = (new_shape[0] - new_unpad[1]) / 2.0
        img = cv2.resize(frame, new_unpad, interpolation=cv2.INTER_LINEAR) if (w, h) != new_unpad else frame
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        if top > 0 or bottom > 0 or left > 0 or right > 0:
            img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return img, r, dw, dh

    def detect(self, frame, conf_thresh=0.25):
        if frame is None or frame.size == 0:
            return []
        h, w = frame.shape[:2]
        lb_img, r, dw, dh = self._letterbox(frame, (640, 640))
        if self.use_ort:
            rgb = cv2.cvtColor(lb_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            inp = np.transpose(rgb, (2, 0, 1))[None, ...]
            preds = self.sess.run([self.output_name], {self.input_name: inp})[0]
            out = preds[0].T
        else:
            blob = cv2.dnn.blobFromImage(lb_img, 1/255.0, (640, 640), swapRB=True, crop=False)
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
        mask = max_scores >= conf_thresh
        if not np.any(mask):
            return []

        v_boxes = boxes[mask]
        filtered_scores = scores[mask]

        candidate_boxes = []
        scores_list = []
        v_cls_chosen = []
        cls_scores_list = []
        check_special = (scores.shape[1] == 5)
        for b, s_row in zip(v_boxes, filtered_scores):
            sub_idx = int(np.argmax(s_row[v_cls]))
            chosen_cid = v_cls[sub_idx]
            chosen_score = float(s_row[chosen_cid])
            # Preserve direct 'car' winner without biased overrides; require clear margins for specialized classes
            v_probs = None
            if check_special and chosen_cid != 1:
                car_s = float(s_row[0])
                amb_s = float(s_row[4])
                auto_s = float(s_row[2])
                bike_s = float(s_row[3])
                v_probs = {'car': car_s, 'auto': auto_s, 'bike': bike_s, 'ambulance': amb_s}
                amb_strong = (amb_s >= 0.65 and amb_s >= car_s * 1.45 and amb_s >= auto_s * 1.45)
                if chosen_cid == 4 and not amb_strong:
                    chosen_cid = 0
                    chosen_score = max(car_s, amb_s * 0.80)
                    v_probs['car'] = max(car_s, chosen_score)
                elif chosen_cid in (2, 3) and ((chosen_score < 0.35 and car_s >= 0.15) or (car_s >= 0.20 and chosen_score < car_s * 1.25)):
                    spec_key = 'auto' if chosen_cid == 2 else 'bike'
                    chosen_cid = 0
                    chosen_score = max(car_s, chosen_score * 0.85)
                    v_probs['car'] = max(car_s, chosen_score)
                    v_probs[spec_key] = min(v_probs[spec_key], max(0.0, float(v_probs['car']) * 0.70))
                if not amb_strong:
                    v_probs['ambulance'] = min(amb_s, max(0.0, float(v_probs['car']) * 0.50))
            elif chosen_cid != 1:
                v_name = normalize_vehicle_type(self.cls_names.get(chosen_cid, 'car'))
                v_probs = {v_name: chosen_score}

            cx, cy, bw, bh = b
            x1 = max(0, int(((cx - bw / 2.0) - dw) / r))
            y1 = max(0, int(((cy - bh / 2.0) - dh) / r))
            x2 = min(w, int(((cx + bw / 2.0) - dw) / r))
            y2 = min(h, int(((cy + bh / 2.0) - dh) / r))
            candidate_boxes.append([x1, y1, max(1, x2 - x1), max(1, y2 - y1)])
            scores_list.append(chosen_score)
            v_cls_chosen.append(chosen_cid)
            cls_scores_list.append(v_probs)

        indices = cv2.dnn.NMSBoxes(candidate_boxes, scores_list, conf_thresh, 0.45)
        results = []
        for i in indices:
            idx = i[0] if isinstance(i, (list, tuple, np.ndarray)) else i
            bx, by, bw, bh = candidate_boxes[idx]
            cid = v_cls_chosen[idx]
            score_val = scores_list[idx]
            cls_scores_val = cls_scores_list[idx]
            if cid == 4 and frame is not None and bw > 10 and bh > 10:
                rx1 = max(0, bx + int(bw * 0.15))
                rx2 = min(w, bx + int(bw * 0.85))
                ry1 = max(0, by + int(bh * 0.12))
                ry2 = min(h, by + int(bh * 0.70))
                body = frame[ry1:ry2, rx1:rx2]
                if body.size > 0:
                    v_chan = cv2.cvtColor(body, cv2.COLOR_BGR2HSV)[:, :, 2]
                    if float(np.percentile(v_chan, 75)) < 130.0:
                        cid = 0
                        if cls_scores_val is not None:
                            cls_scores_val = dict(cls_scores_val)
                            cls_scores_val['car'] = max(float(cls_scores_val.get('car', 0.0)), float(score_val))
                            cls_scores_val['ambulance'] = 0.0
            item = {
                "bbox": [bx, by, bx + bw, by + bh],
                "score": round(score_val, 3),
                "cls": cid,
                "name": self.cls_names.get(cid, 'vehicle')
            }
            if cls_scores_val is not None:
                item["cls_scores"] = cls_scores_val
            results.append(item)

        # Suppress weak partial bumper/sub-box false positives wholly contained inside a high-confidence car
        if len(results) > 1:
            strong_cars = [it for it in results if it['cls'] == 0 and it['score'] >= 0.70]
            if strong_cars:
                kept = []
                for it in results:
                    if it['cls'] in (2, 3, 4) and it['score'] < 0.55:
                        sx1, sy1, sx2, sy2 = it['bbox']
                        s_area = max(1, (sx2 - sx1) * (sy2 - sy1))
                        contained = False
                        for c in strong_cars:
                            cx1, cy1, cx2, cy2 = c['bbox']
                            ix1, iy1 = max(sx1, cx1), max(sy1, cy1)
                            ix2, iy2 = min(sx2, cx2), min(sy2, cy2)
                            if ix2 > ix1 and iy2 > iy1 and ((ix2 - ix1) * (iy2 - iy1) / float(s_area)) >= 0.80:
                                contained = True
                                break
                        if contained:
                            continue
                    kept.append(it)
                results = kept
        return results
def _order_quad(pts):
    """Order any 4 polygon corners clockwise as [top-left, top-right, bottom-right, bottom-left]."""
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    idx_y = np.argsort(pts[:, 1], kind='mergesort')
    top = pts[idx_y[:2]]
    bot = pts[idx_y[2:]]
    top = top[np.argsort(top[:, 0], kind='mergesort')]
    bot = bot[np.argsort(bot[:, 0], kind='mergesort')]
    return np.array([top[0], top[1], bot[1], bot[0]], dtype=np.float32)

class Detector:
    """Multi-stage Hybrid ANPR Pipeline:
       Stage 1: YOLOv8 detects vehicles and direct plates across the full frame.
       Stage 1B: Direct plates from custom YOLO vehicle model (class 'no_plate').
       Stage 2A: Dedicated YuNet License Plate Detector (LPD) across all unmatched vehicles.
       Stage 2B: DBNet text region fallback on foreground vehicles if YuNet misses.
       Stage 3: High-resolution PP-OCRv4 recognition with corner ordering & 2-line support."""
    def __init__(self):
        self.vehicle_detector = VehicleDetector()
        self.db_input_size = 480

        # 1. DBNet text detector (fallback)
        self.detector = cv2.dnn_TextDetectionModel_DB(cv2.dnn.readNet(str(MODELS / 'text_detection_en_ppocrv3_2023may.onnx')))
        self.detector.setBinaryThreshold(0.25)
        self.detector.setPolygonThreshold(0.40)
        self.detector.setUnclipRatio(1.8)
        self.detector.setMaxCandidates(200)
        self.detector.setInputSize((self.db_input_size, self.db_input_size))
        self.detector.setInputMean((123.675, 116.28, 103.53))
        self.detector.setInputScale(1.0 / 255.0 / np.array([0.229, 0.224, 0.225]))

        # 2. Dedicated YuNet License Plate Detector (fixed NMSBoxes + cached priors)
        yunet_path = MODELS / 'license_plate_detection_lpd_yunet_2023mar.onnx'
        self.plate_detector = None
        yunet_cls = MultiPlateYuNet or LPD_YuNet
        if yunet_cls is not None and yunet_path.is_file():
            try:
                self.plate_detector = yunet_cls(modelPath=str(yunet_path), inputSize=[320, 240], confThreshold=0.22)
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
        mask[:int(frame.shape[0] * 0.08), :] = 0
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

    def _infer_ppocr_line(self, img):
        if img is None or img.size == 0:
            return "", 0.0
        ih, iw = img.shape[:2]
        interp = cv2.INTER_CUBIC if (iw < 160 or ih < 32) else cv2.INTER_LINEAR
        resized = cv2.resize(img, (320, 48), interpolation=interp)
        blob = cv2.dnn.blobFromImage(resized, 1.0 / 127.5, (320, 48), (127.5, 127.5, 127.5), swapRB=True)
        self.recognizer_v4.setInput(blob)
        return self._decode_ppocr(self.recognizer_v4.forward())

    @staticmethod
    def _find_two_line_split(crop):
        """Find minimum horizontal gradient energy row near mid-height to split stacked plates cleanly."""
        ph = crop.shape[0]
        lo = max(2, int(ph * 0.36))
        hi = min(ph - 2, int(ph * 0.66))
        if hi <= lo:
            return int(ph * 0.52)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        gx = np.abs(np.diff(gray.astype(np.int16), axis=1))
        row_energy = gx[lo:hi, :].sum(axis=1)
        return lo + int(np.argmin(row_energy))

    def _ocr_detail(self, image, quad):
        ordered = _order_quad(quad)
        w1 = np.linalg.norm(ordered[1] - ordered[0])
        w2 = np.linalg.norm(ordered[2] - ordered[3])
        h1 = np.linalg.norm(ordered[3] - ordered[0])
        h2 = np.linalg.norm(ordered[2] - ordered[1])
        pw = int(max(w1, w2))
        ph = int(max(h1, h2))
        if pw < 12 or ph < 6:
            return "", 0.0

        dst = np.array([[0, 0], [pw - 1, 0], [pw - 1, ph - 1], [0, ph - 1]], dtype=np.float32)
        matrix = cv2.getPerspectiveTransform(ordered, dst)
        crop = cv2.warpPerspective(image, matrix, (pw, ph))
        if crop is None or crop.size == 0:
            return "", 0.0

        if self.use_ppocrv4:
            aspect = pw / max(ph, 1)
            # 2-line stacked plate handling (auto, bike, or square car plates)
            if aspect <= 1.75 and ph >= 18:
                mid_y = self._find_two_line_split(crop)
                t_txt, t_c = self._infer_ppocr_line(crop[:mid_y, :])
                b_txt, b_c = self._infer_ppocr_line(crop[mid_y:, :])
                return t_txt + b_txt, float((t_c + b_c) / 2.0) if (t_txt or b_txt) else 0.0
            text, conf = self._infer_ppocr_line(crop)
            if aspect <= 2.45 and ph >= 18 and not fix_plate(text, ocr_conf=conf):
                mid_y = self._find_two_line_split(crop)
                t_txt, t_c = self._infer_ppocr_line(crop[:mid_y, :])
                b_txt, b_c = self._infer_ppocr_line(crop[mid_y:, :])
                stacked = t_txt + b_txt
                s_conf = float((t_c + b_c) / 2.0) if (t_txt or b_txt) else 0.0
                if fix_plate(stacked, ocr_conf=s_conf):
                    return stacked, s_conf
            return text, conf
        elif self.recognizer_crnn is not None:
            gray = cv2.cvtColor(cv2.resize(crop, (100, 32)), cv2.COLOR_BGR2GRAY)
            self.recognizer_crnn.setInput(cv2.dnn.blobFromImage(gray, size=(100, 32), mean=127.5, scalefactor=1 / 127.5))
            out = self.recognizer_crnn.forward()
            text = ''
            for index in range(out.shape[0]):
                best = int(np.argmax(out[index][0]))
                text += '-' if best == 0 else CHARSET[best - 1]
            cleaned = ''.join(char for i, char in enumerate(text) if char != '-' and (i == 0 or char != text[i - 1]))
            return cleaned, (0.75 if cleaned else 0.0)
        return "", 0.0

    def ocr(self, image, quad):
        return self._ocr_detail(image, quad)[0]

    @staticmethod
    def _match_vehicle(cx, cy, vehicles):
        """Return (index, vehicle_dict) whose bumper/interior best aligns with plate center (cx, cy)."""
        best_idx, best_v, best_rank = None, None, -1e9
        for margin in (0, 6):
            for idx, v in enumerate(vehicles):
                vx1, vy1, vx2, vy2 = v['bbox']
                if (vx1 - margin) <= cx <= (vx2 + margin) and (vy1 - margin) <= cy <= (vy2 + margin):
                    vw = max(1.0, vx2 - vx1)
                    vh = max(1.0, vy2 - vy1)
                    area = vw * vh
                    rel_x = abs(cx - (vx1 + vx2) / 2.0) / (vw * 0.5)
                    rel_y = (cy - vy1) / vh
                    bumper_score = (1.0 - min(1.5, rel_x)) * 1.2 + (0.6 if 0.35 <= rel_y <= 0.98 else 0.0)
                    conf_score = float(v.get('score', 0.5)) * 0.6
                    area_tie = -min(1.0, area / 500000.0) * 0.15
                    rank = bumper_score + conf_score + area_tie - margin * 0.05
                    if rank > best_rank:
                        best_idx, best_v, best_rank = idx, v, rank
            if best_v is not None:
                break
        return best_idx, best_v

    def _try_record_plate(self, frame, quad, det_score, fallback_vehicle, vehicles, seen_quads, detections):
        h, w = frame.shape[:2]
        ordered = _order_quad(quad).astype(np.float64)
        center = ordered.mean(axis=0)
        cx, cy = int(center[0]), int(center[1])
        if is_watermark_zone(cx, cy, w, h):
            return False, None
        if any(np.hypot(center[0] - sc[0], center[1] - sc[1]) < 25 for sc in seen_quads):
            return False, None

        raw_text, ocr_conf = self._ocr_detail(frame, ordered)
        if not raw_text or len(raw_text) < 4 or is_garbage_text(raw_text) or ocr_conf < 0.45:
            return False, None
        normalized = fix_plate(raw_text, ocr_conf=ocr_conf)
        if not normalized:
            return False, None

        m_idx, matched_v = self._match_vehicle(cx, cy, vehicles)
        v_used = matched_v
        if v_used is None and fallback_vehicle is not None:
            fx1, fy1, fx2, fy2 = fallback_vehicle['bbox']
            if (fx1 - 8) <= cx <= (fx2 + 8) and (fy1 - 8) <= cy <= (fy2 + 8):
                v_used = fallback_vehicle
        x_b, y_b, w_b, h_b = cv2.boundingRect(ordered.astype(np.int32))
        area_factor = min(1.0, (w_b * h_b) / 3000.0)
        quality = float((ocr_conf ** 2) * (0.35 + 0.65 * area_factor) * (0.6 + 0.4 * min(1.0, float(det_score))))
        seen_quads.append(center)
        detections.append({
            'plate': normalized.upper(),
            'quad': ordered,
            'bbox': [x_b, y_b, w_b, h_b],
            'score': round(float(det_score), 3),
            'ocr_conf': round(float(ocr_conf), 3),
            'quality': round(quality, 4),
            'vehicle_bbox': v_used['bbox'] if v_used else [x_b, y_b, x_b + w_b, y_b + h_b],
            'vehicle_name': v_used['name'] if v_used else 'vehicle',
            'matched_v': (v_used is not None)
        })
        return True, m_idx

    @staticmethod
    def _box_is_locked(bbox, skip_bboxes):
        if not skip_bboxes:
            return False
        ax1, ay1, ax2, ay2 = bbox
        area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
        for bx1, by1, bx2, by2 in skip_bboxes:
            ix1, iy1 = max(ax1, bx1), max(ay1, by1)
            ix2, iy2 = min(ax2, bx2), min(ay2, by2)
            if ix2 > ix1 and iy2 > iy1:
                inter = (ix2 - ix1) * (iy2 - iy1)
                area_b = max(1, (bx2 - bx1) * (by2 - by1))
                if inter / float(area_a + area_b - inter) >= 0.60:
                    return True
        return False

    def scan(self, frame, skip_bboxes=None):
        h, w = frame.shape[:2]
        # ── STAGE 1: Detect All Vehicles & Direct Plates ───────────────────
        raw_detections = self.vehicle_detector.detect(frame, conf_thresh=0.18)
        if not raw_detections:
            return [], []

        vehicles = [d for d in raw_detections if d['name'] != 'no_plate']
        direct_plates = [d for d in raw_detections if d['name'] == 'no_plate']
        if not vehicles and not direct_plates:
            return [], []

        mask = self.timestamp_mask(frame)
        detections = []
        seen_quads = []
        matched_v_indices = set()

        # Skip OCR for vehicles whose plate is already locked cleanly
        if skip_bboxes:
            for idx, v in enumerate(vehicles):
                if self._box_is_locked(v['bbox'], skip_bboxes):
                    matched_v_indices.add(idx)

        # ── STAGE 1B: Direct License Plate Extraction (from vehicle.onnx) ──
        for p in direct_plates:
            px1, py1, px2, py2 = p['bbox']
            bw, bh = px2 - px1, py2 - py1
            if bw < 18 or bh < 8 or bw > 600 or bh > 200:
                continue
            pcx, pcy = (px1 + px2) / 2.0, (py1 + py2) / 2.0
            m_pre, _ = self._match_vehicle(pcx, pcy, vehicles)
            if m_pre is not None and m_pre in matched_v_indices and skip_bboxes:
                # Only skip if it fell inside a locked track's vehicle box
                if self._box_is_locked(vehicles[m_pre]['bbox'], skip_bboxes):
                    continue
            quad = np.array([[px1, py1], [px2, py1], [px2, py2], [px1, py2]], dtype=np.float64)
            ok_rec, m_idx = self._try_record_plate(frame, quad, p['score'], None, vehicles, seen_quads, detections)
            if not ok_rec:
                # Tight YOLO box fallback retry with slight padding for outer characters
                qx1, qy1 = max(0, px1 - 3), max(0, py1 - 2)
                qx2, qy2 = min(w, px2 + 3), min(h, py2 + 2)
                quad_pad = np.array([[qx1, qy1], [qx2, qy1], [qx2, qy2], [qx1, qy2]], dtype=np.float64)
                ok_rec, m_idx = self._try_record_plate(frame, quad_pad, p['score'], None, vehicles, seen_quads, detections)
            if ok_rec and m_idx is not None:
                matched_v_indices.add(m_idx)

        # ── STAGE 2: Multi-Vehicle Crop ANPR (YuNet LPD + DBNet Fallback) ──
        def vehicle_area(item):
            _, v = item
            vx1, vy1, vx2, vy2 = v['bbox']
            return (vx2 - vx1) * (vy2 - vy1)

        unmatched = [(i, v) for i, v in enumerate(vehicles) if i not in matched_v_indices]
        sorted_vehicles = sorted(unmatched, key=vehicle_area, reverse=True)[:8]
        dbnet_runs = 0

        for v_idx, v in sorted_vehicles:
            vx1, vy1, vx2, vy2 = v['bbox']
            vw, vh = vx2 - vx1, vy2 - vy1
            if vw < 40 or vh < 36:
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

            # Step 2A: Dedicated YuNet License Plate Detector (fixed NMS + ordered corners)
            if self.plate_detector is not None:
                try:
                    self.plate_detector.setInputSize([320, 240])
                    res_yunet = self.plate_detector.infer(cv2.resize(tile, (320, 240)))
                    for yr in res_yunet:
                        pts_tile = yr[:8].reshape(4, 2).copy()
                        pts_tile[:, 0] = pts_tile[:, 0] * (crop_w / 320.0) + x0
                        pts_tile[:, 1] = pts_tile[:, 1] * (crop_h / 240.0) + y0
                        ok_rec, m_idx = self._try_record_plate(frame, pts_tile, float(yr[8]), v, vehicles, seen_quads, detections)
                        if ok_rec:
                            if m_idx is None or m_idx == v_idx:
                                vehicle_plate_found = True
                            if m_idx is not None:
                                matched_v_indices.add(m_idx)
                except Exception:
                    pass

            # Step 2B: Fallback to DBNet only if YuNet didn't read this vehicle's plate
            if not vehicle_plate_found and dbnet_runs < 3 and vw >= 90 and vh >= 70:
                dbnet_runs += 1
                try:
                    boxes, scores = self.detector.detect(cv2.resize(tile, (self.db_input_size, self.db_input_size)))
                except Exception:
                    continue
                if boxes is None:
                    continue

                sx = crop_w / float(self.db_input_size)
                sy = crop_h / float(self.db_input_size)
                for box, score in zip(boxes, scores):
                    quad = np.array(box, dtype=np.float64).reshape(-1, 2)
                    quad[:, 0] = quad[:, 0] * sx + x0
                    quad[:, 1] = quad[:, 1] * sy + y0
                    x, y, bw, bh = cv2.boundingRect(quad.astype(np.int32))
                    if bw < 22 or bh < 8 or bw > 600 or bh > 180 or bw / max(bh, 1) < 1.2 or bw / max(bh, 1) > 9:
                        continue
                    cx, cy = int(x + bw / 2), int(y + bh / 2)
                    pad_w_chk = vw * 0.15
                    pad_h_chk = vh * 0.15
                    if not ((vx1 - pad_w_chk) <= cx <= (vx2 + pad_w_chk) and (vy1 - pad_h_chk) <= cy <= (vy2 + pad_h_chk)):
                        continue
                    if cy < mask.shape[0] and cx < mask.shape[1] and mask[cy, cx] == 0:
                        continue
                    self._try_record_plate(frame, quad, float(score), v, vehicles, seen_quads, detections)
        return detections, vehicles
class Store:
    def __init__(self, path):
        pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.lock = threading.RLock()
        with self.lock:
            self.db.executescript('''
                CREATE TABLE IF NOT EXISTS devices(stream TEXT PRIMARY KEY, name TEXT NOT NULL, role TEXT NOT NULL, anpr INTEGER NOT NULL DEFAULT 1, created_at INTEGER NOT NULL, url TEXT DEFAULT '', masks TEXT DEFAULT '[]', enabled INTEGER NOT NULL DEFAULT 1, gate_group TEXT DEFAULT '', count_enabled INTEGER NOT NULL DEFAULT 1);
                CREATE TABLE IF NOT EXISTS sightings(id TEXT PRIMARY KEY, stream TEXT NOT NULL, plate TEXT NOT NULL, confidence REAL, ts INTEGER NOT NULL, photo BLOB, plate_photo BLOB);
                CREATE INDEX IF NOT EXISTS sight_plate ON sightings(plate, ts);
                CREATE INDEX IF NOT EXISTS sight_ts ON sightings(ts);
                CREATE TABLE IF NOT EXISTS passages(id INTEGER PRIMARY KEY AUTOINCREMENT, stream TEXT NOT NULL, count INTEGER, ts INTEGER NOT NULL);
                CREATE INDEX IF NOT EXISTS passage_ts ON passages(ts);
                CREATE TABLE IF NOT EXISTS vehicle_counts(id TEXT PRIMARY KEY, stream TEXT NOT NULL, direction TEXT NOT NULL, vehicle_type TEXT NOT NULL, plate TEXT, track_id INTEGER, confidence REAL, ts INTEGER NOT NULL, gate_group TEXT DEFAULT '', merged_streams TEXT DEFAULT '');
                CREATE INDEX IF NOT EXISTS vc_dir_ts ON vehicle_counts(direction, ts);
                CREATE INDEX IF NOT EXISTS vc_stream_ts ON vehicle_counts(stream, ts);
                CREATE INDEX IF NOT EXISTS vc_plate_dir ON vehicle_counts(plate, direction, ts);
                CREATE TABLE IF NOT EXISTS ambulance_dispatches(id TEXT PRIMARY KEY, ref_id TEXT, plate TEXT, recipients TEXT NOT NULL, subject TEXT NOT NULL, note TEXT DEFAULT '', message_id TEXT DEFAULT '', ts INTEGER NOT NULL);
                CREATE INDEX IF NOT EXISTS amb_disp_ts ON ambulance_dispatches(ts);
                CREATE INDEX IF NOT EXISTS amb_disp_ref ON ambulance_dispatches(ref_id, ts);
                CREATE TABLE IF NOT EXISTS settings(key TEXT PRIMARY KEY, value TEXT NOT NULL);
            ''')
            for tbl, col, col_def in [
                ("devices", "url", "TEXT DEFAULT ''"),
                ("devices", "masks", "TEXT DEFAULT '[]'"),
                ("devices", "enabled", "INTEGER NOT NULL DEFAULT 1"),
                ("devices", "gate_group", "TEXT DEFAULT ''"),
                ("devices", "count_enabled", "INTEGER NOT NULL DEFAULT 1"),
                ("sightings", "vehicle_type", "TEXT DEFAULT 'car'"),
                ("vehicle_counts", "gate_group", "TEXT DEFAULT ''"),
                ("vehicle_counts", "merged_streams", "TEXT DEFAULT ''"),
            ]:
                try:
                    self.db.execute(f"ALTER TABLE {tbl} ADD COLUMN {col} {col_def}")
                except Exception:
                    pass
            try:
                self.db.execute("CREATE INDEX IF NOT EXISTS vc_dir_grp_ts ON vehicle_counts(direction, gate_group, ts)")
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
                'correlate_window_s': min(3600, max(30, int(rules.get('correlate_window_s', 300)))),
                'gate_dedup_s': min(120, max(0, int(rules.get('gate_dedup_s', 15))))}
    def save_rules(self, body):
        rules = self.rules()
        limits = {'visit_gap_min': (1, 1440), 'exit_grace_min': (1, 1440), 'correlate_window_s': (30, 3600), 'gate_dedup_s': (0, 120)}
        for key, (lo, hi) in limits.items():
            if key in body:
                value = int(body[key])
                if not (lo <= value <= hi):
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
            d['gate_group'] = normalize_gate_group(d.get('gate_group', ''), display=True)
            d['count_enabled'] = bool(d.get('count_enabled', 1))
            result.append(d)
        return result
    def add_device(self, stream, name, role, anpr=True, url='', masks=None, enabled=True, gate_group='', count_enabled=True):
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
            raise ValueError(f"Role must be one of {', '.join(ROLES)}")
        name = str(name or stream).strip()[:80] or stream
        masks_json = json.dumps(masks or [])
        clean_group = normalize_gate_group(gate_group, display=True)
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
            if 'gate_group' in cols:
                data_map['gate_group'] = clean_group
            if 'count_enabled' in cols:
                data_map['count_enabled'] = 1 if count_enabled else 0
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
    def save_sighting(self, stream, plate, confidence, ts, photo=None, plate_photo=None, vehicle_type='car'):
        sid = uuid.uuid4().hex
        vtype = normalize_vehicle_type(vehicle_type)
        with self.lock:
            cols = [r[1] for r in self.db.execute('PRAGMA table_info(sightings)').fetchall()]
            data_map = {'id': sid, 'stream': stream, 'plate': plate, 'confidence': confidence, 'ts': ts, 'photo': photo, 'plate_photo': plate_photo, 'verified': 0, 'vehicle_type': vtype}
            active_cols = [c for c in cols if c in data_map]
            placeholders = ','.join(['?'] * len(active_cols))
            self.db.execute(f"INSERT INTO sightings({','.join(active_cols)}) VALUES({placeholders})", [data_map[c] for c in active_cols])
            self.db.execute('DELETE FROM sightings WHERE ts<?', (ts - 30 * 86400000,))
            self.db.commit()
        return sid
    def update_sighting(self, sid, plate=None, vehicle_type=None, confidence=None):
        """Refine an already-saved sighting in place when multi-frame voting improves plate or type."""
        if not sid:
            return
        with self.lock:
            cols = {r[1] for r in self.db.execute('PRAGMA table_info(sightings)').fetchall()}
            updates, params = [], []
            if plate:
                updates.append('plate=?')
                params.append(plate.upper())
            if vehicle_type and 'vehicle_type' in cols:
                updates.append('vehicle_type=?')
                params.append(normalize_vehicle_type(vehicle_type))
            if confidence is not None:
                updates.append('confidence=?')
                params.append(float(confidence))
            if not updates:
                return
            params.append(sid)
            self.db.execute(f"UPDATE sightings SET {','.join(updates)} WHERE id=?", params)
            self.db.commit()
    def sightings(self, since=0, plate=None, limit=500):
        with self.lock:
            cols = [r[1] for r in self.db.execute('PRAGMA table_info(sightings)').fetchall()]
        has_vtype = 'vehicle_type' in cols
        vtype_sel = ',vehicle_type' if has_vtype else ''
        query = f'SELECT id,stream,plate,confidence,ts,(photo IS NOT NULL),(plate_photo IS NOT NULL){vtype_sel} FROM sightings WHERE ts>=?'
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
                 'plate_url': f'/sightings/{r[0]}/plate' if r[6] else None,
                 'vehicle_type': normalize_vehicle_type(r[7]) if (has_vtype and len(r) > 7 and r[7]) else 'car'} for r in rows]
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
    @staticmethod
    def _pick_upgraded_type(existing_type, candidate_type, existing_conf=0.0, candidate_conf=0.0, trust_candidate=False):
        e = normalize_vehicle_type(existing_type)
        c = normalize_vehicle_type(candidate_type)
        if e == c:
            return e
        if trust_candidate:
            return c
        if {e, c} == {'car', 'ambulance'}:
            ec = float(existing_conf or 0.0)
            cc = float(candidate_conf or 0.0)
            if c == 'ambulance':
                return 'ambulance' if (cc >= 0.70 and cc > ec) else 'car'
            return 'car' if (cc >= ec or ec < 0.70) else 'ambulance'
        if float(candidate_conf or 0.0) > float(existing_conf or 0.0) + 0.08:
            return c
        if e == 'car' and float(existing_conf or 0.0) <= 0.0 and c != 'car':
            return c
        return e
    @staticmethod
    def _are_vtypes_compatible(t1, t2):
        a, b = normalize_vehicle_type(t1), normalize_vehicle_type(t2)
        return a == b or {a, b} <= {'car', 'ambulance'}
    @staticmethod
    def _normalize_plate_set(plates):
        return {re.sub(r'[^A-Z0-9]', '', str(p).upper()) for p in (plates or ()) if p}
    def _claim_gate_row(self, row, stream, vtype, clean_plate=None, conf=None, prefer_new_plate=False):
        """Merge a synchronized multi-camera or track-stitched observation into an existing vehicle_counts record."""
        row_id, row_stream, cur_vtype, cur_plate, cur_conf, cur_merged = row
        merged = _merge_stream_csv(cur_merged or row_stream, row_stream, stream)
        cur_conf_f = float(cur_conf or 0)
        cand_conf_f = float(conf or 0)
        new_vtype = self._pick_upgraded_type(cur_vtype, vtype, cur_conf_f, cand_conf_f)
        if not cur_plate:
            new_plate = clean_plate
        elif clean_plate and prefer_new_plate and cand_conf_f > cur_conf_f + 0.04:
            new_plate = clean_plate
        else:
            new_plate = cur_plate
        new_conf = max(cur_conf_f, cand_conf_f)
        self.db.execute(
            'UPDATE vehicle_counts SET vehicle_type=?, plate=?, confidence=?, merged_streams=? WHERE id=?',
            (new_vtype, new_plate, new_conf, merged, row_id)
        )
        self.db.commit()
        return row_id
    @staticmethod
    def _norm_plate(plate):
        return re.sub(r'[^A-Z0-9]', '', str(plate or '').upper())

    def record_vehicle_count(self, stream, direction, vehicle_type, plate, track_id, confidence, ts, dedup_gap_ms=1800000, gate_group='', gate_dedup_ms=15000, coactive_plates=None):
        """Persist an incoming, outgoing, or basement gate vehicle event with same-gate fuzzy OCR + 1-to-1 dedup."""
        if direction not in GATE_DIRECTIONS:
            return None, False
        vtype = normalize_vehicle_type(vehicle_type)
        clean_plate = plate.upper() if plate else None
        norm_grp = normalize_gate_group(gate_group)
        ts_val = int(ts)
        conf_val = float(confidence or 0)
        gate_dedup_ms = max(0, int(gate_dedup_ms or 0))
        coactive_set = self._normalize_plate_set(coactive_plates)
        with self.lock:
            # Tier 1: Exact or OCR-confusable plate match across the visit gap window
            if clean_plate:
                plate_cands = self.db.execute(
                    'SELECT id, stream, vehicle_type, plate, confidence, merged_streams FROM vehicle_counts '
                    'WHERE direction=? AND plate IS NOT NULL AND ts>=? AND ts<=? ORDER BY ABS(ts - ?) ASC',
                    (direction, ts_val - dedup_gap_ms, ts_val + dedup_gap_ms, ts_val)
                ).fetchall()
                for row in plate_cands:
                    if self._norm_plate(row[3]) in coactive_set:
                        continue
                    if are_plates_similar_vc(row[3], clean_plate, allow_single_edit=False):
                        return self._claim_gate_row(row, stream, vtype, clean_plate, conf_val, prefer_new_plate=True), True

                # Tier 1B: Short-window 1-edit OCR jitter dedup (12s same-camera track split, or gate_dedup_ms peer camera)
                same_cam_win = 12000
                short_win = max(same_cam_win, gate_dedup_ms)
                short_cands = self.db.execute(
                    "SELECT id, stream, vehicle_type, plate, confidence, merged_streams, COALESCE(gate_group,''), ts FROM vehicle_counts "
                    "WHERE direction=? AND plate IS NOT NULL AND ts>=? AND ts<=? ORDER BY ABS(ts - ?) ASC",
                    (direction, ts_val - short_win, ts_val + short_win, ts_val)
                ).fetchall()
                for row in short_cands:
                    if self._norm_plate(row[3]) in coactive_set:
                        continue
                    dt_row = abs(int(row[7]) - ts_val)
                    claimed = set(_merge_stream_csv(row[5] or row[1], row[1]).split(','))
                    in_same_stream = (stream in claimed) and (dt_row <= same_cam_win)
                    in_peer_gate = (stream not in claimed) and (normalize_gate_group(row[6]) == norm_grp) and (dt_row <= gate_dedup_ms)
                    if (in_same_stream or in_peer_gate) and are_plates_similar_vc(row[3], clean_plate, allow_single_edit=True):
                        return self._claim_gate_row(row[:6], stream, vtype, clean_plate, conf_val, prefer_new_plate=True), True

            # Tier 2 & 3: Dynamic multi-camera 1-to-1 consumption within gate_dedup_ms for same (direction, gate_group)
            if gate_dedup_ms > 0:
                gate_rows = self.db.execute(
                    "SELECT id, stream, vehicle_type, plate, confidence, merged_streams, ts FROM vehicle_counts "
                    "WHERE direction=? AND LOWER(TRIM(COALESCE(gate_group,'')))=? AND ts>=? AND ts<=? ORDER BY ABS(ts - ?) ASC",
                    (direction, norm_grp, ts_val - gate_dedup_ms, ts_val + gate_dedup_ms, ts_val)
                ).fetchall()
                unclaimed = []
                for row in gate_rows:
                    claimed = set(_merge_stream_csv(row[5] or row[1], row[1]).split(','))
                    if stream not in claimed:
                        unclaimed.append(row)

                if clean_plate:
                    # Tier 2: 1-char OCR variant on a peer camera at the same gate
                    for row in unclaimed:
                        if row[3] and self._norm_plate(row[3]) not in coactive_set and are_plates_similar_vc(row[3], clean_plate, allow_single_edit=True):
                            return self._claim_gate_row(row[:6], stream, vtype, clean_plate, conf_val, prefer_new_plate=True), True
                    # Tier 3: Attach late plate onto an unplated peer-camera detection of the same vehicle type
                    unplated = [r for r in unclaimed if not r[3] and self._are_vtypes_compatible(r[2], vtype)]
                    unplated.sort(key=lambda r: (0 if normalize_vehicle_type(r[2]) == vtype else 1, abs(int(r[6]) - ts_val)))
                    if unplated:
                        return self._claim_gate_row(unplated[0][:6], stream, vtype, clean_plate, conf_val, prefer_new_plate=True), True
                else:
                    # Tier 3b: Unplated detection on a peer camera claims 1 compatible peer row (prefers exact type + unplated first)
                    compat = [r for r in unclaimed if self._are_vtypes_compatible(r[2], vtype)]
                    compat.sort(key=lambda r: (0 if normalize_vehicle_type(r[2]) == vtype else 1, 0 if not r[3] else 1, abs(int(r[6]) - ts_val)))
                    if compat:
                        return self._claim_gate_row(compat[0][:6], stream, vtype, None, conf_val), True

            eid = uuid.uuid4().hex
            self.db.execute(
                'INSERT INTO vehicle_counts(id, stream, direction, vehicle_type, plate, track_id, confidence, ts, gate_group, merged_streams) VALUES(?,?,?,?,?,?,?,?,?,?)',
                (eid, stream, direction, vtype, clean_plate, track_id, conf_val, ts_val, norm_grp, stream)
            )
            self.db.commit()
        return eid, False
    def update_vehicle_count(self, row_id, vehicle_type=None, plate=None, dedup_gap_ms=1800000, gate_dedup_ms=15000, confidence=None, coactive_plates=None):
        """Refine vehicle type or attach/consensus-update ANPR plate (merging duplicate rows cleanly)."""
        if not row_id:
            return row_id
        gate_dedup_ms = max(0, int(gate_dedup_ms or 0))
        same_cam_win = 12000
        coactive_set = self._normalize_plate_set(coactive_plates)
        with self.lock:
            cur_row = self.db.execute(
                "SELECT stream, direction, COALESCE(gate_group,''), vehicle_type, plate, ts, merged_streams, confidence FROM vehicle_counts WHERE id=?",
                (row_id,)
            ).fetchone()
            if not cur_row:
                return row_id
            cur_stream, direction, cur_grp, cur_vtype, cur_plate, cur_ts, cur_merged, cur_conf = cur_row
            norm_grp = normalize_gate_group(cur_grp)
            cur_streams = [s for s in _merge_stream_csv(cur_merged or cur_stream, cur_stream).split(',') if s]
            cur_stream_set = set(cur_streams)
            cur_conf_f = float(cur_conf or 0)
            cand_conf_f = float(confidence) if confidence is not None else cur_conf_f
            # Allow two-way type correction from track voter when single-camera, or confidence-weighted when merged
            if vehicle_type:
                new_vtype = self._pick_upgraded_type(
                    cur_vtype, vehicle_type, cur_conf_f, cand_conf_f,
                    trust_candidate=(len(cur_streams) <= 1)
                )
            else:
                new_vtype = cur_vtype
            clean_plate = plate.upper() if plate else cur_plate
            if clean_plate and clean_plate != cur_plate:
                cands = self.db.execute(
                    "SELECT id, stream, COALESCE(gate_group,''), vehicle_type, plate, ts, merged_streams, confidence FROM vehicle_counts "
                    "WHERE direction=? AND plate IS NOT NULL AND ts>=? AND ts<=? AND id!=? ORDER BY ABS(ts - ?) ASC",
                    (direction, int(cur_ts) - dedup_gap_ms, int(cur_ts) + dedup_gap_ms, row_id, int(cur_ts))
                ).fetchall()
                other = next((
                    r for r in cands
                    if self._norm_plate(r[4]) not in coactive_set and are_plates_similar_vc(r[4], clean_plate, allow_single_edit=False)
                ), None)
                if not other:
                    for r in cands:
                        if self._norm_plate(r[4]) in coactive_set:
                            continue
                        dt_row = abs(int(r[5]) - int(cur_ts))
                        other_set = set(_merge_stream_csv(r[6] or r[1], r[1]).split(','))
                        in_same_stream = (not cur_stream_set.isdisjoint(other_set)) and (dt_row <= same_cam_win)
                        in_peer_gate = cur_stream_set.isdisjoint(other_set) and (normalize_gate_group(r[2]) == norm_grp) and (dt_row <= gate_dedup_ms)
                        if (in_same_stream or in_peer_gate) and are_plates_similar_vc(r[4], clean_plate, allow_single_edit=True):
                            other = r
                            break
                if other:
                    other_conf_f = float(other[7] or 0)
                    target_type = self._pick_upgraded_type(other[3], new_vtype, other_conf_f, cand_conf_f)
                    merged_union = _merge_stream_csv(other[6] or other[1], other[1], *cur_streams)
                    best_plate = clean_plate if (cand_conf_f > other_conf_f + 0.04) else other[4]
                    best_conf = max(other_conf_f, cand_conf_f)
                    self.db.execute(
                        'UPDATE vehicle_counts SET vehicle_type=?, plate=?, confidence=?, merged_streams=? WHERE id=?',
                        (target_type, best_plate, best_conf, merged_union, other[0])
                    )
                    self.db.execute('DELETE FROM vehicle_counts WHERE id=?', (row_id,))
                    self.db.commit()
                    return other[0]
            new_conf = max(cur_conf_f, cand_conf_f)
            self.db.execute(
                'UPDATE vehicle_counts SET vehicle_type=?, plate=?, confidence=?, merged_streams=? WHERE id=?',
                (new_vtype, clean_plate, new_conf, ','.join(cur_streams), row_id)
            )
            self.db.commit()
        return row_id
    @staticmethod
    def _empty_vtype_bucket():
        return {'car': 0, 'bike': 0, 'auto': 0, 'ambulance': 0, 'total': 0}

    @staticmethod
    def _append_direction_filter(sql, params, direction):
        if direction == 'main':
            sql += " AND direction IN ('incoming','outgoing')"
        elif direction == 'basement':
            sql += " AND direction IN ('basement_incoming','basement_outgoing')"
        elif direction in GATE_DIRECTIONS:
            sql += ' AND direction=?'
            params.append(direction)
        return sql

    def vehicle_counts_summary(self, from_ts=0, to_ts=None, granularity='hourly', event_limit=50, direction=None, vehicle_type=None):
        """Query persisted historical incoming/outgoing and basement vehicle counts from vehicle_counts table."""
        end_ts = int(to_ts) if to_ts is not None else int(time.time() * 1000)
        start_ts = max(0, int(from_ts or 0))
        devices = {d['stream']: d for d in self.devices()}
        ev_sql = (
            "SELECT id, stream, direction, vehicle_type, plate, confidence, ts, COALESCE(gate_group,''), COALESCE(merged_streams,'') "
            "FROM vehicle_counts WHERE ts>=? AND ts<=?"
        )
        ev_params = [start_ts, end_ts]
        ev_sql = self._append_direction_filter(ev_sql, ev_params, direction)
        if vehicle_type in VEHICLE_TYPES:
            ev_sql += ' AND vehicle_type=?'
            ev_params.append(vehicle_type)
        ev_sql += ' ORDER BY ts DESC LIMIT ?'
        ev_params.append(max(1, int(event_limit)))

        with self.lock:
            agg_rows = self.db.execute(
                'SELECT direction, vehicle_type, COUNT(*) FROM vehicle_counts WHERE ts>=? AND ts<=? GROUP BY direction, vehicle_type',
                (start_ts, end_ts)
            ).fetchall()
            cam_rows = self.db.execute(
                'SELECT stream, direction, vehicle_type, COUNT(*) FROM vehicle_counts WHERE ts>=? AND ts<=? GROUP BY stream, direction, vehicle_type',
                (start_ts, end_ts)
            ).fetchall()
            raw_hist = self.db.execute(
                'SELECT direction, vehicle_type, ts FROM vehicle_counts WHERE ts>=? AND ts<=? ORDER BY ts',
                (start_ts, end_ts)
            ).fetchall()
            event_rows = self.db.execute(ev_sql, ev_params).fetchall()
            all_rows = self.db.execute(
                'SELECT direction, vehicle_type, COUNT(*) FROM vehicle_counts GROUP BY direction, vehicle_type'
            ).fetchall()
            bounds = self.db.execute('SELECT MIN(ts), MAX(ts) FROM vehicle_counts').fetchone()

        agg_buckets = {d: self._empty_vtype_bucket() for d in GATE_DIRECTIONS}
        for dir_key, vtype, cnt in agg_rows:
            if dir_key in agg_buckets:
                vt = normalize_vehicle_type(vtype)
                agg_buckets[dir_key][vt] += int(cnt)
                agg_buckets[dir_key]['total'] += int(cnt)

        all_buckets = {d: self._empty_vtype_bucket() for d in GATE_DIRECTIONS}
        for dir_key, vtype, cnt in all_rows:
            if dir_key in all_buckets:
                vt = normalize_vehicle_type(vtype)
                all_buckets[dir_key][vt] += int(cnt)
                all_buckets[dir_key]['total'] += int(cnt)

        incoming = agg_buckets['incoming']
        outgoing = agg_buckets['outgoing']
        all_in = all_buckets['incoming']
        all_out = all_buckets['outgoing']

        cam_map = {}
        for stream, dir_key, vtype, cnt in cam_rows:
            vt = normalize_vehicle_type(vtype)
            key = (stream, dir_key)
            if key not in cam_map:
                dev = devices.get(stream, {})
                cam_map[key] = {
                    'stream': stream,
                    'name': dev.get('name', stream),
                    'role': dev.get('role', dir_key),
                    'direction': dir_key,
                    'gate_group': dev.get('gate_group', ''),
                    **self._empty_vtype_bucket()
                }
            cam_map[key][vt] += int(cnt)
            cam_map[key]['total'] += int(cnt)

        if granularity in ('hourly', 'daily'):
            is_daily = (granularity == 'daily')
        else:
            is_daily = (start_ts > 0 and (end_ts - start_ts) > 48 * 3600000) or (start_ts == 0)
        bucket_ms = 86400000 if is_daily else 3600000
        hist_buckets = {}
        basement_hist_buckets = {}
        for dir_key, vtype, ts_val in raw_hist:
            if dir_key in ('incoming', 'outgoing'):
                target_map, side_key = hist_buckets, dir_key
            elif dir_key in ('basement_incoming', 'basement_outgoing'):
                target_map = basement_hist_buckets
                side_key = 'incoming' if dir_key == 'basement_incoming' else 'outgoing'
            else:
                continue
            vt = normalize_vehicle_type(vtype)
            b_ts = (int(ts_val) + IST_OFFSET_MS) // bucket_ms * bucket_ms - IST_OFFSET_MS
            slot = target_map.setdefault(b_ts, {
                'bucket': b_ts,
                'incoming': self._empty_vtype_bucket(),
                'outgoing': self._empty_vtype_bucket()
            })
            side = slot[side_key]
            side[vt] += 1
            side['total'] += 1

        recent_events = []
        for r in event_rows:
            streams_seen = [s for s in _merge_stream_csv(r[8] or r[1], r[1]).split(',') if s]
            dev_info = devices.get(r[1], {})
            grp_disp = dev_info.get('gate_group') or r[7] or ''
            recent_events.append({
                'id': r[0],
                'stream': r[1],
                'camera': dev_info.get('name', r[1]),
                'direction': r[2],
                'vehicle_type': normalize_vehicle_type(r[3]),
                'plate': r[4],
                'confidence': round(float(r[5]), 2) if r[5] is not None else None,
                'ts': r[6],
                'gate_group': grp_disp,
                'merged_streams': streams_seen,
                'merged_cameras': [devices.get(s, {}).get('name', s) for s in streams_seen]
            })

        return {
            'window': {'from_ts': start_ts, 'to_ts': end_ts, 'granularity': 'daily' if is_daily else 'hourly'},
            'incoming': incoming,
            'outgoing': outgoing,
            'basement': {
                'incoming': agg_buckets['basement_incoming'],
                'outgoing': agg_buckets['basement_outgoing'],
                'history': [basement_hist_buckets[k] for k in sorted(basement_hist_buckets.keys())],
                'all_time': {
                    'incoming': all_buckets['basement_incoming'],
                    'outgoing': all_buckets['basement_outgoing'],
                }
            },
            'by_camera': sorted(cam_map.values(), key=lambda x: (x['direction'], -x['total'])),
            'history': [hist_buckets[k] for k in sorted(hist_buckets.keys())],
            'recent_events': recent_events,
            'all_time': {
                'incoming': all_in,
                'outgoing': all_out,
                'earliest_ts': bounds[0] if bounds else None,
                'latest_ts': bounds[1] if bounds else None
            }
        }
    def vehicle_counts_events(self, from_ts=0, to_ts=None, direction=None, vehicle_type=None, limit=5000):
        end_ts = int(to_ts) if to_ts is not None else int(time.time() * 1000)
        start_ts = max(0, int(from_ts or 0))
        devices = {d['stream']: d for d in self.devices()}
        sql = "SELECT id, stream, direction, vehicle_type, plate, confidence, ts, COALESCE(gate_group,''), COALESCE(merged_streams,'') FROM vehicle_counts WHERE ts>=? AND ts<=?"
        params = [start_ts, end_ts]
        sql = self._append_direction_filter(sql, params, direction)
        if vehicle_type in VEHICLE_TYPES:
            sql += ' AND vehicle_type=?'
            params.append(vehicle_type)
        sql += ' ORDER BY ts DESC LIMIT ?'
        params.append(max(1, min(20000, int(limit))))
        with self.lock:
            rows = self.db.execute(sql, params).fetchall()
        out = []
        for r in rows:
            streams_seen = [s for s in _merge_stream_csv(r[8] or r[1], r[1]).split(',') if s]
            dev_info = devices.get(r[1], {})
            out.append({
                'id': r[0], 'stream': r[1], 'camera': dev_info.get('name', r[1]),
                'direction': r[2], 'vehicle_type': normalize_vehicle_type(r[3]),
                'plate': r[4], 'confidence': round(float(r[5]), 2) if r[5] is not None else None, 'ts': r[6],
                'gate_group': dev_info.get('gate_group') or r[7] or '',
                'merged_streams': streams_seen,
                'merged_cameras': [devices.get(s, {}).get('name', s) for s in streams_seen]
            })
        return out
def _pick_journey_plate(reads):
    scores = collections.defaultdict(float)
    for r in reads:
        p = str(r.get('plate') or '').upper()
        if not p:
            continue
        bonus = 0.05 if STRICT_RE.match(p.lower()) else 0.0
        scores[p] += float(r.get('confidence') or 0.70) + bonus
    return max(scores.items(), key=lambda x: x[1])[0] if scores else (reads[0].get('plate') or '')

def _pick_journey_vtype(reads):
    weights = collections.defaultdict(float)
    counts = collections.Counter()
    for r in reads:
        vt = normalize_vehicle_type(r.get('vehicle_type'))
        weights[vt] += max(0.30, float(r.get('confidence') or 0.70))
        counts[vt] += 1
    total = sum(weights.values()) or 1.0
    amb_share = weights.get('ambulance', 0.0) / total
    req_amb = 2 if len(reads) >= 2 else 1
    if counts.get('ambulance', 0) >= req_amb and amb_share > 0.55:
        return 'ambulance'
    non_amb = [(vt, w) for vt, w in weights.items() if vt != 'ambulance']
    if non_amb:
        return max(non_amb, key=lambda x: x[1])[0]
    return max(weights.items(), key=lambda x: x[1])[0]

def journeys(store, since_ms, plate_filter=None):
    """Cluster plate sightings into visits tolerating OCR confusables and single-char passage jitter."""
    rules = store.rules()
    gap_ms = rules['visit_gap_min'] * 60000
    grace_ms = rules['exit_grace_min'] * 60000
    devices = {d['stream']: d for d in store.devices()}
    # Load window sightings (fetch unfiltered when plate_filter given so 1-char variants cluster before filtering)
    rows = store.sightings(since=since_ms, plate=None, limit=20000)
    now_ms = int(time.time() * 1000)

    visits = []
    # Index clusters by normalized confusable key for O(1) exact/confusable matching
    exact_buckets = collections.defaultdict(list)
    for read in rows:
        plate = str(read.get('plate') or '').upper()
        if not plate:
            continue
        conf_key = re.sub(r'[^A-Z0-9]', '', plate).translate(_VC_CONFUSABLE)
        ts = int(read['ts'])
        dev = devices.get(read['stream'], {})
        role = dev.get('role', 'checkpoint')

        matched_visit = None
        for v in reversed(exact_buckets.get(conf_key, ())):
            if ts - v['last_ts'] <= gap_ms:
                matched_visit = v
                break

        if matched_visit is None:
            # Fallback: short-window same-camera (<= 25s) or cross-camera open-visit 1-edit match
            for v in reversed(visits):
                dt = ts - v['last_ts']
                if dt > gap_ms:
                    break
                if not any(are_plates_similar_vc(vp, plate, allow_single_edit=True) for vp in v['plates']):
                    continue
                same_stream_jitter = (read['stream'] == v['last_stream'] and dt <= 25000)
                cross_cam_visit = (read['stream'] != v['last_stream'] and dt <= grace_ms and v['last_role'] != 'exit')
                if same_stream_jitter or cross_cam_visit:
                    matched_visit = v
                    break

        if matched_visit is None:
            matched_visit = {
                'reads': [],
                'plates': set(),
                'conf_keys': set(),
                'last_ts': ts,
                'last_stream': read['stream'],
                'last_role': role
            }
            visits.append(matched_visit)

        matched_visit['reads'].append(read)
        matched_visit['plates'].add(plate)
        matched_visit['last_ts'] = ts
        matched_visit['last_stream'] = read['stream']
        matched_visit['last_role'] = role
        if conf_key not in matched_visit['conf_keys']:
            matched_visit['conf_keys'].add(conf_key)
            exact_buckets[conf_key].append(matched_visit)

    out = []
    norm_filter = plate_filter.strip().upper() if plate_filter else None
    for visit in visits:
        reads_v = visit['reads']
        plate = _pick_journey_plate(reads_v)
        if norm_filter and norm_filter not in visit['plates'] and not any(
            are_plates_similar_vc(vp, norm_filter, allow_single_edit=False) for vp in visit['plates']
        ):
            continue
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
        j_vtype = _pick_journey_vtype(reads_v)
        out.append({'plate': plate, 'vehicle_type': j_vtype, 'status': status, 'exit_confirmed': exit_confirmed,
                    'entry': {'id': first.get('id'), 'stream': first['stream'], 'device': devices.get(first['stream'], {}).get('name', first['stream']), 'at': first['ts'], 'photo_url': first['photo_url'], 'plate_url': first['plate_url'], 'confidence': first['confidence']},
                    'exit': {'id': last.get('id'), 'stream': last['stream'], 'device': devices.get(last['stream'], {}).get('name', last['stream']), 'at': exited_at, 'photo_url': last['photo_url'], 'plate_url': last['plate_url'], 'confidence': last['confidence']} if exited_at else None,
                    'cameras': cameras, 'camera_count': len(cameras), 'reads': len(reads_v),
                    'first_at': first['ts'], 'last_at': last['ts'], 'dwell_ms': max(0, end - first['ts'])})
    out.sort(key=lambda v: v['first_at'], reverse=True)
    return out

def record_ambulance_dispatch(store, ref_id, plate, recipients, subject, note='', message_id='', ts=None):
    did = uuid.uuid4().hex
    ts_val = int(ts if ts is not None else time.time() * 1000)
    clean_plate = plate.upper() if plate else ''
    recips_str = ', '.join(recipients) if isinstance(recipients, (list, tuple)) else str(recipients or '')
    with store.lock:
        store.db.execute(
            'INSERT INTO ambulance_dispatches(id, ref_id, plate, recipients, subject, note, message_id, ts) VALUES(?,?,?,?,?,?,?,?)',
            (did, ref_id or '', clean_plate, recips_str, subject, note or '', message_id or '', ts_val)
        )
        store.db.commit()
    return {'id': did, 'ref_id': ref_id or '', 'plate': clean_plate, 'recipients': recips_str, 'subject': subject, 'note': note or '', 'message_id': message_id or '', 'ts': ts_val}

def list_ambulance_dispatches(store, since_ms=0, limit=500):
    with store.lock:
        rows = store.db.execute(
            'SELECT id, ref_id, plate, recipients, subject, note, message_id, ts FROM ambulance_dispatches WHERE ts>=? ORDER BY ts DESC LIMIT ?',
            (max(0, int(since_ms or 0)), max(1, int(limit)))
        ).fetchall()
    return [
        {'id': r[0], 'ref_id': r[1], 'plate': r[2], 'recipients': r[3], 'subject': r[4], 'note': r[5], 'message_id': r[6], 'ts': r[7]}
        for r in rows
    ]

def build_ambulance_list(store, hours=24, limit=200):
    """Build unified Ambulance Registry combining plated ambulance journeys and unplated gate counts."""
    now_ms = int(time.time() * 1000)
    since_ms = int(now_ms - hours * 3600000)
    amb_journeys = [v for v in journeys(store, since_ms) if v.get('vehicle_type') == 'ambulance']
    vc_events = store.vehicle_counts_events(from_ts=since_ms, to_ts=now_ms, vehicle_type='ambulance', limit=1000)
    dispatches = list_ambulance_dispatches(store, since_ms=since_ms, limit=500)

    by_ref = {}
    by_plate_disp = {}
    for d in dispatches:
        if d.get('ref_id') and d['ref_id'] not in by_ref:
            by_ref[d['ref_id']] = d
        if d.get('plate') and d['plate'] not in by_plate_disp:
            by_plate_disp[d['plate']] = d

    items = []
    for v in amb_journeys:
        entry = v.get('entry') or {}
        exit_info = v.get('exit')
        item_id = entry.get('id') or f"j_{v['plate']}_{v['first_at']}"
        last_disp = by_ref.get(item_id) or by_plate_disp.get(v['plate'])
        items.append({
            'id': item_id,
            'source': 'journey',
            'plate': v['plate'],
            'status': v['status'],
            'exit_confirmed': bool(v.get('exit_confirmed')),
            'direction': 'exited' if v['status'] == 'exited' else 'on_site',
            'entry': entry,
            'exit': exit_info,
            'dwell_ms': v.get('dwell_ms', 0),
            'first_at': v['first_at'],
            'last_at': v['last_at'],
            'cameras': v.get('cameras', []),
            'reads': v.get('reads', 1),
            'confidence': max([c.get('confidence') or 0 for c in v.get('cameras', [])] or [entry.get('confidence') or 0]),
            'photo_url': (exit_info or {}).get('photo_url') or entry.get('photo_url'),
            'plate_url': entry.get('plate_url') or (exit_info or {}).get('plate_url'),
            'entry_sighting_id': entry.get('id'),
            'exit_sighting_id': (exit_info or {}).get('id'),
            'last_dispatch': last_disp,
        })

    # Include unplated (or standalone gate-count) ambulances not covered by a plated journey
    for ev in vc_events:
        ev_plate = ev.get('plate')
        ev_ts = int(ev['ts'])
        covered = False
        if ev_plate:
            for it in items:
                if it['source'] == 'journey' and are_plates_similar_vc(it['plate'], ev_plate, allow_single_edit=True):
                    if (it['first_at'] - 120000) <= ev_ts <= (it['last_at'] + 120000):
                        covered = True
                        break
        if covered:
            continue
        is_in = (ev.get('direction') in ('incoming', 'basement_incoming'))
        cam_node = {
            'stream': ev['stream'],
            'name': ev.get('camera') or ev['stream'],
            'role': 'entry' if is_in else 'exit',
            'first': ev_ts,
            'last': ev_ts,
            'reads': 1,
            'confidence': ev.get('confidence')
        }
        items.append({
            'id': ev['id'],
            'source': 'gate_count',
            'plate': ev_plate,
            'status': 'on_site' if is_in else 'exited',
            'exit_confirmed': not is_in,
            'direction': ev.get('direction'),
            'entry': {'stream': ev['stream'], 'device': ev.get('camera'), 'at': ev_ts, 'confidence': ev.get('confidence')} if is_in else None,
            'exit': {'stream': ev['stream'], 'device': ev.get('camera'), 'at': ev_ts, 'confidence': ev.get('confidence')} if not is_in else None,
            'dwell_ms': 0,
            'first_at': ev_ts,
            'last_at': ev_ts,
            'cameras': [cam_node],
            'merged_cameras': ev.get('merged_cameras') or [ev.get('camera') or ev['stream']],
            'reads': 1,
            'confidence': ev.get('confidence') or 0,
            'photo_url': None,
            'plate_url': None,
            'entry_sighting_id': None,
            'exit_sighting_id': None,
            'last_dispatch': by_ref.get(ev['id']) or (by_plate_disp.get(ev_plate) if ev_plate else None),
        })

    items.sort(key=lambda x: x.get('last_at') or x.get('first_at') or 0, reverse=True)
    trimmed = items[:max(1, int(limit))]
    return {
        'ambulances': trimmed,
        'summary': {
            'total': len(items),
            'on_site': sum(1 for x in items if x['status'] == 'on_site'),
            'exited': sum(1 for x in items if x['status'] == 'exited'),
            'plated': sum(1 for x in items if x.get('plate')),
            'unplated': sum(1 for x in items if not x.get('plate')),
            'dispatched': sum(1 for x in items if x.get('last_dispatch')),
        }
    }

def build_ambulance_email_html(subject, title=None, summary=None, metadata=None, status_tag='on_site', has_plate_crop=False, has_vehicle_photo=False):
    """Build custom Medanta Ambulance Alert HTML compatible with 31040 inline CID attachments."""
    is_active = (str(status_tag or '').lower() in ('on_site', 'incoming', 'arrival'))
    badge_bg = '#fee2e2' if is_active else '#e0f2fe'
    badge_color = '#b91c1c' if is_active else '#0369a1'
    badge_border = '#fca5a5' if is_active else '#7dd3fc'
    badge_text = 'AMBULANCE ON PREMISES' if is_active else 'AMBULANCE EXITED'
    accent_color = '#dc2626' if is_active else '#0284c7'

    safe_subject = html.escape(str(subject or 'Medanta Ambulance Alert'))
    safe_title = html.escape(str(title or '🚑 Ambulance Gate Movement Advisory'))
    summary_html = ''
    if summary:
        summary_html = (
            f'<div style="margin-bottom:20px;padding:12px 16px;background-color:#f8fafc;'
            f'border-left:4px solid {accent_color};border-radius:4px;font-size:14px;color:#334155;line-height:1.5;">'
            f'{html.escape(str(summary))}</div>'
        )

    meta_rows = ''
    for k, v in (metadata or {}).items():
        if v is None or not str(v).strip():
            continue
        k_esc = html.escape(str(k))
        v_str = str(v)
        v_esc = html.escape(v_str)
        v_low = v_str.lower()
        if any(w in v_low for w in ('on premises', 'still on premises', 'emergency', 'urgent', 'priority')):
            val_style = 'color:#b91c1c;font-weight:700;background-color:#fee2e2;padding:2px 6px;border-radius:4px;display:inline-block;'
        elif any(w in v_low for w in ('exited', 'departed', 'cleared')):
            val_style = 'color:#0369a1;font-weight:700;background-color:#e0f2fe;padding:2px 6px;border-radius:4px;display:inline-block;'
        else:
            val_style = 'color:#0f172a;font-weight:600;'
        meta_rows += (
            '<tr style="border-bottom:1px solid #f1f5f9;">'
            f'<td style="padding:9px 12px;font-size:13px;color:#64748b;font-weight:600;width:38%;vertical-align:top;">{k_esc}</td>'
            f'<td style="padding:9px 12px;font-size:13px;vertical-align:top;"><span style="{val_style}">{v_esc}</span></td>'
            '</tr>'
        )

    evidence_section = ''
    if has_plate_crop or has_vehicle_photo:
        cols = []
        if has_plate_crop:
            cols.append(
                '<td align="center" style="background-color:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:14px;vertical-align:top;">'
                '<div style="font-size:11px;font-weight:700;color:#475569;text-transform:uppercase;margin-bottom:8px;">🪪 License Plate Crop</div>'
                '<img src="cid:face_crop" alt="License Plate Crop" style="max-height:110px;max-width:100%;object-fit:contain;border-radius:6px;border:2px solid #cbd5e1;display:block;margin:0 auto;" />'
                '<div style="font-size:10px;color:#94a3b8;margin-top:6px;">ANPR Plate Read</div></td>'
            )
        if has_vehicle_photo:
            cols.append(
                '<td align="center" style="background-color:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:14px;vertical-align:top;">'
                '<div style="font-size:11px;font-weight:700;color:#475569;text-transform:uppercase;margin-bottom:8px;">🚑 Ambulance Vehicle Snapshot</div>'
                '<img src="cid:body_crop" alt="Ambulance Snapshot" style="max-height:200px;max-width:100%;object-fit:contain;border-radius:6px;border:2px solid #cbd5e1;display:block;margin:0 auto;" />'
                '<div style="font-size:10px;color:#94a3b8;margin-top:6px;">Gate Camera Capture</div></td>'
            )
        evidence_section = (
            '<div style="margin-top:24px;padding-top:20px;border-top:1px solid #e2e8f0;">'
            '<div style="font-size:13px;font-weight:800;color:#1e293b;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:14px;">'
            '📷 Ambulance Visual Evidence</div>'
            f'<table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="border-collapse:separate;border-spacing:10px 0;"><tr>{"".join(cols)}</tr></table>'
            '</div>'
        )

    return f'''<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0"><title>{safe_subject}</title></head>
<body style="margin:0;padding:0;background-color:#f1f5f9;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;color:#0f172a;">
<table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="background-color:#f1f5f9;padding:24px 12px;">
<tr><td align="center">
<table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="max-width:620px;background-color:#ffffff;border-radius:12px;overflow:hidden;box-shadow:0 4px 6px -1px rgba(0,0,0,0.1);border:1px solid #e2e8f0;">
<tr><td style="background-color:#0f172a;padding:16px 24px;color:#ffffff;">
<table role="presentation" width="100%" cellpadding="0" cellspacing="0"><tr>
<td><div style="font-size:11px;text-transform:uppercase;letter-spacing:1.2px;color:#38bdf8;font-weight:700;">🏥 MEDANTA HOSPITAL OPERATIONS</div>
<div style="font-size:16px;font-weight:800;color:#ffffff;margin-top:2px;">Ambulance &amp; Emergency Gate Tracker</div></td>
<td align="right" style="vertical-align:middle;"><span style="display:inline-block;padding:4px 10px;border-radius:9999px;font-size:10px;font-weight:800;background-color:{badge_bg};color:{badge_color};border:1px solid {badge_border};text-transform:uppercase;">{badge_text}</span></td>
</tr></table></td></tr>
<tr><td style="padding:24px 24px 20px 24px;">
<div style="font-size:18px;font-weight:800;color:#0f172a;margin-bottom:12px;">{safe_title}</div>
{summary_html}
<div style="border:1px solid #e2e8f0;border-radius:8px;overflow:hidden;margin-top:14px;">
<div style="background-color:#f8fafc;padding:8px 12px;font-size:11px;font-weight:700;color:#64748b;text-transform:uppercase;letter-spacing:0.5px;border-bottom:1px solid #e2e8f0;">Ambulance Movement Details</div>
<table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="border-collapse:collapse;">{meta_rows}</table>
</div>
{evidence_section}
</td></tr>
<tr><td style="background-color:#f8fafc;padding:16px 24px;border-top:1px solid #e2e8f0;text-align:center;">
<div style="font-size:11px;color:#64748b;line-height:1.4;">Dispatched via <strong>Medanta Hospital AI Ops Suite</strong> • Ambulance &amp; Vehicle Tracker.</div>
<div style="font-size:10px;color:#94a3b8;margin-top:4px;">GoDezk AI Vision Fleet • Powered by AWS SES</div>
</td></tr></table></td></tr></table></body></html>'''

def relay_email_dispatch(payload):
    """Send email dispatch JSON payload to centralized 31040 AWS SES microservice."""
    raw_body = json.dumps(payload).encode('utf-8')
    last_net_err = None
    urls = [u.strip() for u in os.environ.get('EMAIL_DISPATCH_URL', '').split(',') if u.strip()] or EMAIL_DISPATCH_URLS
    for url in urls:
        req = urllib.request.Request(url, data=raw_body, headers={'Content-Type': 'application/json'}, method='POST')
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                return json.loads(resp.read().decode('utf-8'))
        except urllib.error.HTTPError as http_err:
            err_body = http_err.read().decode('utf-8', errors='replace')
            try:
                detail = json.loads(err_body).get('detail', err_body)
            except Exception:
                detail = err_body or str(http_err)
            raise RuntimeError(f'Email service error ({http_err.code}): {detail}')
        except (urllib.error.URLError, OSError, TimeoutError) as net_err:
            last_net_err = net_err
            continue
    raise RuntimeError(f'Could not reach Email Dispatch service on port 31040: {last_net_err}')
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
                    'tracks': {}, 'retired_tracks': {}, 'next_track_id': 1,
                    'live_counts': {
                        'in_frame': Store._empty_vtype_bucket(),
                        'session': Store._empty_vtype_bucket(),
                        'updated_at': None
                    },
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
    def canonical(self, worker, plate, score=0.85, center=None):
        """Merge same-vehicle OCR jitter in pending/seen buffers without collapsing co-present vehicles."""
        for other, item in worker['pending'].items():
            if not are_plates_similar_vc(other, plate, allow_single_edit=True):
                continue
            if center is not None and item.get('center') is not None:
                if float(np.linalg.norm(center - item['center'])) > 260 and not are_plates_similar_vc(other, plate, allow_single_edit=False):
                    continue
            cands = item.setdefault('candidates', {})
            cands[plate] = cands.get(plate, 0.0) + float(score or 0.5)
            return max(cands.items(), key=lambda x: x[1])[0]
        return plate
    @staticmethod
    def _bbox_iou(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        inter = (ix2 - ix1) * (iy2 - iy1)
        area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
        area_b = max(1, (bx2 - bx1) * (by2 - by1))
        return inter / float(area_a + area_b - inter)

    @staticmethod
    def _track_edge_clipped(bbox, fw, fh):
        x1, y1, x2, y2 = bbox
        return x1 <= 4 or y1 <= 4 or x2 >= (fw - 4) or y2 >= (fh - 4)

    @staticmethod
    def _resolve_track_type(type_votes):
        if not type_votes:
            return 'car'
        best_t = max(VEHICLE_TYPES, key=lambda t: (type_votes.get(t, 0), 1 if t == 'car' else 0))
        return best_t if type_votes.get(best_t, 0) > 0 else 'car'

    def _update_track_type_votes(self, tr, dv, fw, fh):
        clipped = self._track_edge_clipped(dv['bbox'], fw, fh)
        tr['last_edge_clipped'] = clipped
        vw = max(1, dv['bbox'][2] - dv['bbox'][0])
        vh = max(1, dv['bbox'][3] - dv['bbox'][1])
        spatial_w = 0.55 if clipped else (1.15 if (vw * vh) >= 15000 else 1.0)
        type_scores = tr.setdefault('type_scores', {})
        cls_scores = dv.get('cls_scores')
        if cls_scores:
            for vt, sc in cls_scores.items():
                nvt = normalize_vehicle_type(vt)
                type_scores[nvt] = type_scores.get(nvt, 0.0) + float(sc) * spatial_w
        else:
            vt = dv['vtype']
            type_scores[vt] = type_scores.get(vt, 0.0) + float(dv.get('score', 0.5)) * spatial_w
        tr['types'][dv['vtype']] += 1

    def _resolve_track_type_from_track(self, tr):
        type_scores = tr.get('type_scores')
        if not type_scores:
            return self._resolve_track_type(tr.get('types'))
        types_cnt = tr.get('types') or {}
        amb_cnt = int(types_cnt.get('ambulance', 0))
        car_cnt = int(types_cnt.get('car', 0))
        amb_w = float(type_scores.get('ambulance', 0.0))
        car_w = float(type_scores.get('car', 0.0))
        allow_amb = (
            amb_cnt >= 3
            and amb_cnt > car_cnt
            and amb_w >= max(1.90, car_w * 1.35)
        )
        cand_types = VEHICLE_TYPES if allow_amb else ('car', 'auto', 'bike')
        best_type = max(cand_types, key=lambda t: (type_scores.get(t, 0.0), 1 if t == 'car' else 0))
        if best_type in ('auto', 'bike') and car_w > 0.0:
            bt_cnt = int(types_cnt.get(best_type, 0))
            bt_w = float(type_scores.get(best_type, 0.0))
            if bt_cnt < 2 and bt_w < max(0.45, car_w * 1.15):
                best_type = 'car'
        cur = tr.get('resolved_type')
        if cur and best_type != cur and tr.get('type_confirmed'):
            if cur == 'ambulance' and not allow_amb:
                tr['resolved_type'] = best_type
                return best_type
            cur_w = type_scores.get(cur, 0.0)
            best_w = type_scores.get(best_type, 0.0)
            best_cnt = types_cnt.get(best_type, 0)
            if cur == 'ambulance' and best_type == 'car':
                margin = 1.15
            elif best_type == 'ambulance':
                margin = 1.35
            else:
                margin = 1.40
            if best_cnt < 2 or best_w < cur_w * margin:
                return cur
        tr['resolved_type'] = best_type
        return best_type

    def _add_track_plate_vote(self, tr, det, now_mono):
        plate = str(det['plate']).upper()
        q = float(det.get('quality', max(0.05, (det.get('ocr_conf', 0.75) ** 2) * 0.8)))
        if STRICT_RE.match(plate.lower()):
            q *= 1.15
        votes = tr.setdefault('plate_votes', {})
        counts = tr.setdefault('plate_counts', {})
        pos_votes = tr.setdefault('plate_pos_votes', {})
        votes[plate] = votes.get(plate, 0.0) + q
        counts[plate] = counts.get(plate, 0) + 1
        slots = pos_votes.setdefault(len(plate), [collections.defaultdict(float) for _ in range(len(plate))])
        for i, ch in enumerate(plate):
            slots[i][ch] += q

        if q >= tr.get('best_plate_q', 0.0):
            tr['best_plate_q'] = q
            tr['best_plate_det'] = det
            tr['best_plate_mono'] = now_mono
            tx1, ty1 = tr['bbox'][0], tr['bbox'][1]
            tr['best_plate_offset'] = det['quad'] - np.array([tx1, ty1], dtype=np.float64)
            tr['best_crop_stale'] = True

        self._resolve_track_plate(tr)

    def _resolve_track_plate(self, tr):
        votes = tr.get('plate_votes')
        counts = tr.get('plate_counts')
        pos_votes = tr.get('plate_pos_votes')
        if not votes:
            return None

        top_whole, _ = max(votes.items(), key=lambda x: x[1])
        consensus = top_whole

        if pos_votes:
            best_len = max(pos_votes.items(), key=lambda kv: sum(max(d.values()) for d in kv[1]))[0]
            char_str = ''.join(max(slot.items(), key=lambda x: x[1])[0] for slot in pos_votes[best_len])
            if char_str != top_whole and counts.get(top_whole, 0) < 2 and STRICT_RE.match(char_str.lower()):
                fixed_char = fix_plate(char_str, ocr_conf=0.90)
                if fixed_char == char_str and are_plates_similar_vc(char_str, top_whole, allow_single_edit=True):
                    consensus = char_str

        cluster_cnt = sum(c for k, c in counts.items() if are_plates_similar_vc(k, consensus, allow_single_edit=True))
        cluster_w = sum(w for k, w in votes.items() if are_plates_similar_vc(k, consensus, allow_single_edit=True))
        unrelated_w = sum(w for k, w in votes.items() if not are_plates_similar_vc(k, consensus, allow_single_edit=True))

        best_det = tr.get('best_plate_det') or {}
        best_ocr = float(best_det.get('ocr_conf', 0.0))
        pb = best_det.get('bbox') or [0, 0, 0, 0]
        pw = pb[2] if len(pb) >= 3 else 0

        confirmed = (cluster_cnt >= 2) or (best_ocr >= 0.85 and pw >= 70 and bool(STRICT_RE.match(consensus.lower())))
        locked = confirmed and (cluster_cnt >= 3 or (cluster_cnt >= 2 and best_ocr >= 0.88)) and (cluster_w >= max(0.45, unrelated_w * 1.5))

        cur_plate = tr.get('plate')
        if cur_plate and tr.get('plate_locked'):
            cur_w = sum(w for k, w in votes.items() if are_plates_similar_vc(k, cur_plate, allow_single_edit=True))
            if not are_plates_similar_vc(consensus, cur_plate, allow_single_edit=True) and cluster_w <= cur_w * 1.35:
                consensus = cur_plate
            elif are_plates_similar_vc(consensus, cur_plate, allow_single_edit=True) and cluster_w <= votes.get(cur_plate, 0.0) * 1.20:
                if counts.get(cur_plate, 0) >= 2:
                    consensus = cur_plate

        if confirmed:
            tr['plate'] = consensus
            tr['plate_confirmed'] = True
        if locked:
            tr['plate_locked'] = True
        return tr.get('plate')

    def _update_tracks_and_collect_ops(self, worker, stream, role, count_enabled, vehicles, detections, frame_shape, now_mono, now_ms):
        """Match detected vehicles to persistent per-camera tracks with motion prediction, stitching, and multi-frame voting."""
        fh, fw = frame_shape[:2]
        valid_vehicles = []
        for v in vehicles:
            vx1, vy1, vx2, vy2 = v['bbox']
            vw, vh = vx2 - vx1, vy2 - vy1
            if v.get('score', 0) < 0.24 or vw < 34 or vh < 34 or (vw * vh) < 1600:
                continue
            cx, cy = (vx1 + vx2) / 2.0, (vy1 + vy2) / 2.0
            if is_watermark_zone(int(cx), int(cy), fw, fh):
                continue
            vtype = normalize_vehicle_type(v.get('name'))
            valid_vehicles.append({
                'bbox': [vx1, vy1, vx2, vy2],
                'center': (cx, cy),
                'area': float(vw * vh),
                'score': float(v.get('score', 0)),
                'vtype': vtype,
                'cls_scores': v.get('cls_scores'),
                'diag': float(np.hypot(vw, vh)),
                'matched_dets': []
            })

        # Assign frame's plate detections to best enclosing vehicle detections for track stitching & voting
        orphan_dets = []
        for det in detections:
            dc = det['quad'].mean(axis=0)
            has_vbox = bool(det.get('matched_v', True))
            best_di, best_dm = None, 0.0
            for di, dv in enumerate(valid_vehicles):
                vx1, vy1, vx2, vy2 = dv['bbox']
                vw, vh = max(1.0, vx2 - vx1), max(1.0, vy2 - vy1)
                strict_in = vx1 <= dc[0] <= vx2 and vy1 <= dc[1] <= vy2
                loose_in = (vx1 - 6) <= dc[0] <= (vx2 + 6) and (vy1 - 6) <= dc[1] <= (vy2 + 6)
                if not loose_in and not has_vbox:
                    continue
                iou = self._bbox_iou(dv['bbox'], det['vehicle_bbox']) if (has_vbox and det.get('vehicle_bbox')) else 0.0
                if not loose_in and iou < 0.35:
                    continue
                rel_x = abs(dc[0] - (vx1 + vx2) / 2.0) / (vw * 0.5)
                rel_y = (dc[1] - vy1) / vh
                bumper = (0.30 if 0.35 <= rel_y <= 0.98 else 0.0) + max(0.0, 1.0 - min(1.5, rel_x)) * 0.35
                m_val = iou + (0.65 if strict_in else (0.30 if loose_in else 0.0)) + bumper
                if m_val > best_dm:
                    best_dm, best_di = m_val, di
            if best_di is not None and best_dm > 0.15:
                valid_vehicles[best_di]['matched_dets'].append(det)
                det['_track_di'] = best_di
            else:
                orphan_dets.append(det)

        tracks = worker.setdefault('tracks', {})
        retired = worker.setdefault('retired_tracks', {})

        # Retire tracks unseen > 8.0s into a 12.0s stitching buffer; purge stale retired tracks
        for tid in [k for k, tr in tracks.items() if now_mono - tr['last_mono'] > 8.0]:
            retired[tid] = tracks.pop(tid)
        for tid in [k for k, tr in retired.items() if now_mono - tr['last_mono'] > 12.0]:
            retired.pop(tid, None)
        if len(retired) > 25:
            for tid, _ in sorted(retired.items(), key=lambda kv: kv[1]['last_mono'])[:-25]:
                retired.pop(tid, None)

        # Greedy 1-to-1 matching against active tracks using velocity prediction + size sanity
        pairs = []
        for tid, tr in tracks.items():
            dt = max(0.0, min(2.0, now_mono - tr['last_mono']))
            vx, vy = tr.get('vel', (0.0, 0.0))
            tx1, ty1, tx2, ty2 = tr['bbox']
            dx_p, dy_p = vx * dt, vy * dt
            pred_bbox = [tx1 + dx_p, ty1 + dy_p, tx2 + dx_p, ty2 + dy_p]
            pred_cx, pred_cy = tr['center'][0] + dx_p, tr['center'][1] + dy_p
            tdiag = float(np.hypot(tx2 - tx1, ty2 - ty1))
            t_area = max(1.0, (tx2 - tx1) * (ty2 - ty1))
            for di, dv in enumerate(valid_vehicles):
                iou = max(self._bbox_iou(tr['bbox'], dv['bbox']), self._bbox_iou(pred_bbox, dv['bbox']))
                area_ratio = dv['area'] / t_area
                if (area_ratio < 0.30 or area_ratio > 3.3) and iou < 0.40:
                    continue
                dist_raw = float(np.hypot(tr['center'][0] - dv['center'][0], tr['center'][1] - dv['center'][1]))
                dist_pred = float(np.hypot(pred_cx - dv['center'][0], pred_cy - dv['center'][1]))
                dist = min(dist_raw, dist_pred)
                max_diag = max(tdiag, dv['diag'], 1.0)
                if iou >= 0.20 or dist <= 0.55 * max_diag:
                    type_pen = 0.0 if Store._are_vtypes_compatible(tr.get('resolved_type', dv['vtype']), dv['vtype']) else 0.15
                    score = iou + max(0.0, 1.0 - dist / max_diag) * 0.45 - type_pen
                    pairs.append((score, tid, di))
        pairs.sort(reverse=True)

        matched_tids, matched_didx = set(), set()
        for _, tid, di in pairs:
            if tid in matched_tids or di in matched_didx:
                continue
            matched_tids.add(tid)
            matched_didx.add(di)
            valid_vehicles[di]['_tid'] = tid
            self._apply_det_to_track(tracks[tid], valid_vehicles[di], fh, fw, now_mono)

        for tid, tr in tracks.items():
            if tid not in matched_tids:
                tr['misses'] += 1

        # Track Stitching: attempt to reconnect remaining detections to recently retired tracks before allocating new IDs
        active_plates = {tr['plate'] for tr in tracks.values() if tr.get('plate') and tr.get('misses', 0) == 0}
        for di, dv in enumerate(valid_vehicles):
            if di in matched_didx:
                continue
            stitched_tid = self._try_stitch_retired(dv, retired, active_plates, now_mono)
            if stitched_tid is not None:
                tr = retired.pop(stitched_tid)
                tracks[stitched_tid] = tr
                self._apply_det_to_track(tr, dv, fh, fw, now_mono)
                matched_didx.add(di)
                matched_tids.add(stitched_tid)
                dv['_tid'] = stitched_tid
                continue

            tid = worker.get('next_track_id', 1)
            worker['next_track_id'] = tid + 1
            tr = {
                'id': tid,
                'bbox': dv['bbox'],
                'center': dv['center'],
                'vel': (0.0, 0.0),
                'first_ts': now_ms,
                'last_mono': now_mono,
                'hits': 1,
                'misses': 0,
                'best_score': dv['score'],
                'score_sum': dv['score'],
                'types': collections.Counter(),
                'type_scores': {},
                'resolved_type': dv['vtype'],
                'type_confirmed': False,
                'last_edge_clipped': self._track_edge_clipped(dv['bbox'], fw, fh),
                'plate': None,
                'plate_votes': {},
                'plate_counts': {},
                'plate_pos_votes': {},
                'plate_confirmed': False,
                'plate_locked': False,
                'best_plate_det': None,
                'best_plate_q': 0.0,
                'best_plate_offset': None,
                'counted': False,
                'db_id': None,
                'saved_type': None,
                'saved_plate': None,
                'sighting_id': None,
                'sighting_plate': None,
                'sighting_vtype': None,
                'sighting_suppressed': False
            }
            self._update_track_type_votes(tr, dv, fw, fh)
            tr['resolved_type'] = self._resolve_track_type_from_track(tr)
            for det in dv['matched_dets']:
                self._add_track_plate_vote(tr, det, now_mono)
            tracks[tid] = tr
            dv['_tid'] = tid

        # Track plate->tid mapping on matched detections
        for di, dv in enumerate(valid_vehicles):
            if '_tid' not in dv:
                for tid, tr in tracks.items():
                    if tr['misses'] == 0 and tr['bbox'] == dv['bbox']:
                        dv['_tid'] = tid
                        break
            tid = dv.get('_tid')
            if tid is not None and tid in tracks:
                for det in dv['matched_dets']:
                    det['_tid'] = tid

        in_frame = Store._empty_vtype_bucket()
        for tid, tr in tracks.items():
            if tr['misses'] == 0 and (tr['hits'] >= 2 or tr['best_score'] >= 0.35):
                vt = tr.get('resolved_type') or 'car'
                in_frame[vt] += 1
                in_frame['total'] += 1

        direction = COUNTABLE_ROLE_DIRECTION.get(role) if count_enabled else None
        db_ops = []

        # Collect coactive confirmed plates to protect distinct adjacent fleet vehicles
        visible_plates_by_tid = {
            tid: tr['plate'] for tid, tr in tracks.items()
            if tr.get('plate') and tr.get('misses', 0) == 0
        }

        if direction in GATE_DIRECTIONS:
            for tid, tr in tracks.items():
                avg_score = tr.get('score_sum', tr['best_score']) / max(1, tr['hits'])
                is_confirmed = (
                    tr['hits'] >= 3
                    or (tr['hits'] >= 2 and not tr.get('last_edge_clipped', False) and avg_score >= 0.45)
                    or (tr['hits'] >= 2 and (tr.get('plate_confirmed') or tr['best_score'] >= 0.55))
                )
                if not is_confirmed:
                    continue
                tr['type_confirmed'] = True
                v_type = self._resolve_track_type_from_track(tr)
                coactive = tuple(p for other_tid, p in visible_plates_by_tid.items() if other_tid != tid)
                if not tr['counted']:
                    tr['counted'] = True
                    tr['saved_type'] = v_type
                    tr['saved_plate'] = tr['plate']
                    db_ops.append(('insert', tid, direction, v_type, tr['plate'], tr['best_score'], tr['first_ts'], coactive))
                elif tr['db_id'] and (v_type != tr['saved_type'] or (tr['plate'] and tr['plate'] != tr['saved_plate'])):
                    tr['saved_type'] = v_type
                    if tr['plate']:
                        tr['saved_plate'] = tr['plate']
                    db_ops.append(('update', tid, tr['db_id'], v_type, tr['plate'], tr['best_score'], coactive))
        else:
            # Remaining cameras (or gate cameras with count_enabled=False): update live-only session & in-frame counts without DB writes
            live = worker.setdefault('live_counts', {
                'in_frame': Store._empty_vtype_bucket(),
                'session': Store._empty_vtype_bucket(),
                'updated_at': None
            })
            sess = live['session']
            for tid, tr in tracks.items():
                avg_score = tr.get('score_sum', tr['best_score']) / max(1, tr['hits'])
                is_confirmed = (
                    tr['hits'] >= 3
                    or (tr['hits'] >= 2 and not tr.get('last_edge_clipped', False) and avg_score >= 0.45)
                    or (tr['hits'] >= 2 and (tr.get('plate_confirmed') or tr['best_score'] >= 0.55))
                )
                if not is_confirmed:
                    continue
                tr['type_confirmed'] = True
                v_type = self._resolve_track_type_from_track(tr)
                if not tr['counted']:
                    tr['counted'] = True
                    tr['saved_type'] = v_type
                    sess[v_type] += 1
                    sess['total'] += 1
                elif v_type != tr['saved_type']:
                    old_t = tr['saved_type']
                    if old_t and sess.get(old_t, 0) > 0:
                        sess[old_t] -= 1
                        sess[v_type] += 1
                    tr['saved_type'] = v_type
            live['in_frame'] = in_frame
            live['updated_at'] = now_ms

        return db_ops

    def _apply_det_to_track(self, tr, dv, fh, fw, now_mono):
        dt = now_mono - tr['last_mono']
        if 0.05 < dt < 2.5:
            inst_vx = (dv['center'][0] - tr['center'][0]) / dt
            inst_vy = (dv['center'][1] - tr['center'][1]) / dt
            ovx, ovy = tr.get('vel', (0.0, 0.0))
            tr['vel'] = (0.6 * ovx + 0.4 * inst_vx, 0.6 * ovy + 0.4 * inst_vy)
        tr['bbox'] = dv['bbox']
        tr['center'] = dv['center']
        tr['last_mono'] = now_mono
        tr['hits'] += 1
        tr['misses'] = 0
        tr['best_score'] = max(tr['best_score'], dv['score'])
        tr['score_sum'] = tr.get('score_sum', 0.0) + dv['score']
        self._update_track_type_votes(tr, dv, fw, fh)
        self._resolve_track_type_from_track(tr)
        for det in dv['matched_dets']:
            self._add_track_plate_vote(tr, det, now_mono)

    def _try_stitch_retired(self, dv, retired, active_plates, now_mono):
        if not retired:
            return None
        dv_plates = [d['plate'] for d in dv.get('matched_dets', ()) if d.get('plate')]
        best_tid, best_rank = None, -1.0
        for tid, rtr in retired.items():
            dt = now_mono - rtr['last_mono']
            if dt > 12.0:
                continue
            r_plate = rtr.get('plate')
            if dv_plates and r_plate and r_plate not in active_plates:
                if any(are_plates_similar_vc(dp, r_plate, allow_single_edit=True) for dp in dv_plates):
                    rank = 10.0 - dt * 0.2
                    if rank > best_rank:
                        best_rank, best_tid = rank, tid
                    continue
            if dt <= 5.0 and Store._are_vtypes_compatible(rtr.get('resolved_type', dv['vtype']), dv['vtype']):
                rx1, ry1, rx2, ry2 = rtr['bbox']
                r_area = max(1.0, (rx2 - rx1) * (ry2 - ry1))
                ratio = dv['area'] / r_area
                if 0.45 <= ratio <= 2.2:
                    vx, vy = rtr.get('vel', (0.0, 0.0))
                    pred_cx = rtr['center'][0] + vx * min(2.0, dt)
                    pred_cy = rtr['center'][1] + vy * min(2.0, dt)
                    rdiag = float(np.hypot(rx2 - rx1, ry2 - ry1))
                    dist = float(np.hypot(pred_cx - dv['center'][0], pred_cy - dv['center'][1]))
                    max_d = max(rdiag, dv['diag'], 1.0)
                    if dist <= 0.70 * max_d:
                        rank = 2.0 - dist / max_d
                        if rank > best_rank:
                            best_rank, best_tid = rank, tid
        return best_tid

    def remaining_cameras_live(self):
        """Return live in-memory vehicle counts for non-gate cameras or gate cameras with counting disabled (not saved to DB)."""
        devices = [d for d in self.store.devices() if d.get('role') not in COUNTABLE_ROLE_DIRECTION or not d.get('count_enabled', True)]
        out = []
        with self.lock:
            for d in devices:
                s = d['stream']
                w = self.workers.get(s)
                live = (w or {}).get('live_counts') or {}
                meta = (w or {}).get('meta') or {}
                in_frame = dict(live.get('in_frame') or Store._empty_vtype_bucket())
                session = dict(live.get('session') or Store._empty_vtype_bucket())
                online = bool(w and w.get('jpeg') is not None and not meta.get('error'))
                out.append({
                    'stream': s,
                    'name': d.get('name', s),
                    'role': d.get('role', 'checkpoint'),
                    'count_enabled': bool(d.get('count_enabled', True)),
                    'gate_group': d.get('gate_group', ''),
                    'online': online,
                    'error': meta.get('error'),
                    'updated_at': live.get('updated_at'),
                    'in_frame': in_frame,
                    'session': session
                })
        return out

    @staticmethod
    def _encode_sighting_crops(image, vbox, pbox):
        ih, iw = image.shape[:2]
        x, y, pw, ph = pbox
        crop = None
        if vbox:
            vx1, vy1, vx2, vy2 = vbox
            crop = image[max(0, vy1 - 15):min(ih, vy2 + 15), max(0, vx1 - 15):min(iw, vx2 + 15)]
        if crop is None or crop.size == 0:
            margin = int(max(pw, ph) * 1.4)
            crop = image[max(0, y - margin):min(ih, y + ph + margin), max(0, x - margin):min(iw, x + pw + margin)]
        if crop is None or crop.size == 0:
            return None, None
        ok, photo = cv2.imencode('.jpg', crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            return None, None
        plate_crop = image[max(0, y - 8):min(ih, y + ph + 8), max(0, x - 8):min(iw, x + pw + 8)]
        plate_img = None
        if plate_crop is not None and plate_crop.size > 0:
            ok2, enc_p = cv2.imencode('.jpg', plate_crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if ok2:
                plate_img = enc_p.tobytes()
        return photo.tobytes(), plate_img

    def infer_loop(self):
        last_frame = {}
        last_infer_time = {}
        while True:
            now = time.monotonic()
            with self.lock:
                items = [(stream, w['latest'], w) for stream, w in self.workers.items()]
            # Temporal Decimation: pace inference to ~5 FPS (200ms interval) per camera, oldest-scanned first
            items = [(s, latest, w) for s, latest, w in items
                     if latest and last_frame.get(s) != latest[1] and (now - last_infer_time.get(s, 0)) >= 0.20]
            if not items:
                time.sleep(0.03)
                continue
            items.sort(key=lambda item: last_infer_time.get(item[0], 0))
            dev_map = {d['stream']: d for d in self.store.devices()}
            rules_cfg = self.store.rules()
            dedup_gap_ms = rules_cfg.get('visit_gap_min', 30) * 60000
            gate_dedup_ms = rules_cfg.get('gate_dedup_s', 15) * 1000
            for stream, (original, captured), worker in items:
                last_frame[stream] = captured
                last_infer_time[stream] = time.monotonic()
                detector = self.ensure_detector()
                if detector is None:
                    time.sleep(1)
                    break
                frame = original.copy()
                try:
                    with self.lock:
                        skip_bboxes = [
                            tr['bbox'] for tr in worker.get('tracks', {}).values()
                            if tr.get('plate_locked') and tr.get('misses', 0) == 0 and (tr.get('hits', 0) % 5 != 0)
                        ]
                    with self.model_lock:
                        scan_start = time.monotonic()
                        detections, vehicles = detector.scan(frame, skip_bboxes=skip_bboxes)
                        scan_ms = (time.monotonic() - scan_start) * 1000

                    now = time.monotonic()
                    now_ms = int(time.time() * 1000)
                    stale_crops = []
                    sighting_ops = []
                    hud_vehicles = []
                    hud_detections = []

                    with self.lock:
                        worker['seen'] = {p: ts for p, ts in worker['seen'].items() if now - ts < 45}
                        worker['pending'] = {p: i for p, i in worker['pending'].items() if now - i['last'] < 20}

                        dev_cfg = dev_map.get(stream, {})
                        dev_role = (dev_cfg.get('role') or 'checkpoint').lower()
                        dev_count_enabled = bool(dev_cfg.get('count_enabled', True))
                        dev_gate_group = dev_cfg.get('gate_group', '')
                        track_db_ops = self._update_tracks_and_collect_ops(
                            worker, stream, dev_role, dev_count_enabled,
                            vehicles, detections, frame.shape, now, now_ms
                        )
                        tracks = worker.get('tracks', {})
                        visible_plates = {
                            tid: tr['plate'] for tid, tr in tracks.items()
                            if tr.get('plate') and tr.get('misses', 0) == 0
                        }
                        fresh_plate_tids = {det['_tid'] for det in detections if '_tid' in det}

                        for tid, tr in tracks.items():
                            if tr.get('best_crop_stale') and tr.get('best_plate_det') and not tr.get('sighting_id'):
                                stale_crops.append((tid, tr['bbox'], tr['best_plate_det']['bbox']))
                                tr['best_crop_stale'] = False
                            if tr['misses'] == 0 and (tr['hits'] >= 2 or tr['best_score'] >= 0.42):
                                hud_vehicles.append({
                                    'bbox': tr['bbox'],
                                    'name': tr.get('resolved_type', 'car'),
                                    'score': tr['best_score']
                                })
                                if tid not in fresh_plate_tids and tr.get('plate_locked') and tr.get('plate') and tr.get('best_plate_offset') is not None:
                                    tx1, ty1 = tr['bbox'][0], tr['bbox'][1]
                                    quad = tr['best_plate_offset'] + np.array([tx1, ty1], dtype=np.float64)
                                    xb, yb, wb, hb = cv2.boundingRect(quad.astype(np.int32))
                                    hud_detections.append({
                                        'plate': tr['plate'],
                                        'quad': quad,
                                        'bbox': [xb, yb, wb, hb]
                                    })

                            if tr.get('plate_confirmed') and tr.get('plate') and tr['misses'] == 0:
                                plate = tr['plate']
                                v_type = tr.get('resolved_type', 'car')
                                coactive = {p for o_tid, p in visible_plates.items() if o_tid != tid}
                                best_det = tr.get('best_plate_det') or {}
                                det_score = float(best_det.get('score', tr['best_score']))
                                ready_for_sighting = (tr['hits'] >= 2)
                                if ready_for_sighting and not tr.get('sighting_id') and not tr.get('sighting_suppressed'):
                                    seen_match = next(
                                        (sp for sp in worker['seen'] if sp not in coactive and are_plates_similar_vc(sp, plate, allow_single_edit=True)),
                                        None
                                    )
                                    if seen_match:
                                        worker['seen'][seen_match] = now
                                        tr['sighting_suppressed'] = True
                                    else:
                                        worker['seen'][plate] = now
                                        tr['sighting_plate'] = plate
                                        tr['sighting_vtype'] = v_type
                                        sighting_ops.append((
                                            'insert', tid, plate, det_score, now_ms,
                                            tr['bbox'], best_det.get('bbox'),
                                            tr.get('best_photo_jpeg'), tr.get('best_plate_jpeg'), v_type
                                        ))
                                elif tr.get('sighting_id') and (plate != tr.get('sighting_plate') or v_type != tr.get('sighting_vtype')):
                                    worker['seen'][plate] = now
                                    tr['sighting_plate'] = plate
                                    tr['sighting_vtype'] = v_type
                                    sighting_ops.append(('update', tid, tr['sighting_id'], plate, v_type, det_score))

                        # Display track consensus plate on live HUD detections + fallback for rare unboxed orphan plates
                        for det in detections:
                            tid_m = det.get('_tid')
                            disp_plate = det['plate']
                            if tid_m in tracks and tracks[tid_m].get('plate'):
                                disp_plate = tracks[tid_m]['plate']
                            hud_detections.append({
                                'plate': disp_plate,
                                'quad': det['quad'],
                                'bbox': det['bbox']
                            })
                            if tid_m is not None:
                                continue

                            center = det['quad'].mean(axis=0)
                            can_plate = self.canonical(worker, det['plate'], det.get('ocr_conf', det['score']), center=center)
                            seen_orph = next(
                                (sp for sp in worker['seen'] if sp not in visible_plates.values() and are_plates_similar_vc(sp, can_plate, allow_single_edit=True)),
                                None
                            )
                            if seen_orph:
                                worker['seen'][seen_orph] = now
                                continue
                            pend_key = next(
                                (pk for pk in worker['pending'] if are_plates_similar_vc(pk, can_plate, allow_single_edit=True)),
                                can_plate
                            )
                            previous = worker['pending'].get(pend_key)
                            sightings = (previous['count'] if previous else 0) + 1
                            cands = (previous.get('candidates') if previous else {}) or {can_plate: det.get('ocr_conf', det['score'])}
                            cands[det['plate']] = cands.get(det['plate'], 0.0) + det.get('ocr_conf', det['score'])
                            worker['pending'][pend_key] = {'count': sightings, 'center': center, 'last': now, 'candidates': cands}
                            ocr_conf = det.get('ocr_conf', 0.0)
                            if sightings < 2 and ocr_conf < 0.85:
                                continue
                            worker['seen'][can_plate] = now
                            sighting_ops.append((
                                'insert_orphan', can_plate, det['score'], now_ms,
                                det.get('vehicle_bbox'), det['bbox'],
                                normalize_vehicle_type(det.get('vehicle_name'))
                            ))

                        worker['meta'].update(scan_ms=round(scan_ms, 1), plates=len(hud_detections),
                                              frame_age_ms=round((time.monotonic() - captured) * 1000))
                        self.stats['scans'] += 1

                    # Encode crops & execute SQLite writes outside engine.lock so MJPEG streaming never stalls
                    cached_jpegs = {}
                    for tid, vbox, pbox in stale_crops:
                        ph_b, pl_b = self._encode_sighting_crops(original, vbox, pbox)
                        if ph_b:
                            cached_jpegs[tid] = (ph_b, pl_b)
                    if cached_jpegs:
                        with self.lock:
                            tmap = self.workers.get(stream, {}).get('tracks', {})
                            for tid, (ph_b, pl_b) in cached_jpegs.items():
                                if tid in tmap and not tmap[tid].get('sighting_id'):
                                    tmap[tid]['best_photo_jpeg'] = ph_b
                                    tmap[tid]['best_plate_jpeg'] = pl_b

                    for sop in sighting_ops:
                        if sop[0] == 'insert':
                            _, tid, plate, conf, ts_ms, vbox, pbox, ph_b, pl_b, v_type = sop
                            if tid in cached_jpegs:
                                ph_b, pl_b = cached_jpegs[tid]
                            elif ph_b is None and pbox is not None:
                                ph_b, pl_b = self._encode_sighting_crops(original, vbox, pbox)
                            if ph_b:
                                sid = self.store.save_sighting(stream, plate, conf, ts_ms, ph_b, pl_b, v_type)
                                self.stats['sightings'] += 1
                                with self.lock:
                                    tr_cur = self.workers.get(stream, {}).get('tracks', {}).get(tid)
                                    if tr_cur:
                                        tr_cur['sighting_id'] = sid
                                        tr_cur['best_photo_jpeg'] = None
                                        tr_cur['best_plate_jpeg'] = None
                        elif sop[0] == 'update':
                            _, tid, sid, plate, v_type, conf = sop
                            self.store.update_sighting(sid, plate=plate, vehicle_type=v_type, confidence=conf)
                        elif sop[0] == 'insert_orphan':
                            _, plate, conf, ts_ms, vbox, pbox, v_type = sop
                            ph_b, pl_b = self._encode_sighting_crops(original, vbox, pbox)
                            if ph_b:
                                self.store.save_sighting(stream, plate, conf, ts_ms, ph_b, pl_b, v_type)
                                self.stats['sightings'] += 1

                    for op in track_db_ops:
                        if op[0] == 'insert':
                            _, tid, direction, v_type, plate, conf, first_ts, coactive = op
                            row_id, _ = self.store.record_vehicle_count(
                                stream, direction, v_type, plate, tid, conf, first_ts,
                                dedup_gap_ms=dedup_gap_ms, gate_group=dev_gate_group, gate_dedup_ms=gate_dedup_ms,
                                coactive_plates=coactive
                            )
                            with self.lock:
                                if stream in self.workers and tid in self.workers[stream].get('tracks', {}):
                                    self.workers[stream]['tracks'][tid]['db_id'] = row_id
                        elif op[0] == 'update':
                            _, tid, row_id, v_type, plate, conf, coactive = op
                            new_id = self.store.update_vehicle_count(
                                row_id, v_type, plate, dedup_gap_ms=dedup_gap_ms, gate_dedup_ms=gate_dedup_ms,
                                confidence=conf, coactive_plates=coactive
                            )
                            if new_id != row_id:
                                with self.lock:
                                    if stream in self.workers and tid in self.workers[stream].get('tracks', {}):
                                        self.workers[stream]['tracks'][tid]['db_id'] = new_id

                    with self.lock:
                        if stream in self.workers:
                            worker['active_boxes'] = (hud_vehicles, hud_detections)
                except Exception as err:
                    print(f"[ANPR error] {stream}: {err}", flush=True)
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
                vtype_filter = (query.get('type', [None])[0] or query.get('vehicle_type', [None])[0] or '').strip().lower()
                result = journeys(self.store, int(time.time() * 1000 - hours * 3600000), plate)
                if vtype_filter and vtype_filter != 'all':
                    norm_vt = normalize_vehicle_type(vtype_filter)
                    result = [v for v in result if normalize_vehicle_type(v.get('vehicle_type')) == norm_vt]
                return self.send_json(200, {'journeys': result, 'count': len(result)})
            if path_str in ('/journeys.csv', '/journeys/export'):
                hours = min(720, max(1, float(query.get('hours', ['24'])[0])))
                plate = query.get('plate', [None])[0]
                status_filter = query.get('status', [None])[0]
                camera_filter = query.get('camera', [None])[0]
                vtype_filter = (query.get('type', [None])[0] or query.get('vehicle_type', [None])[0] or '').strip().lower()
                result = journeys(self.store, int(time.time() * 1000 - hours * 3600000), plate)
                if status_filter:
                    result = [v for v in result if v['status'] == status_filter]
                if vtype_filter and vtype_filter != 'all':
                    norm_vt = normalize_vehicle_type(vtype_filter)
                    result = [v for v in result if normalize_vehicle_type(v.get('vehicle_type')) == norm_vt]
                if camera_filter:
                    result = [v for v in result if any(c['stream'] == camera_filter for c in v['cameras'])
                              or (v.get('entry') or {}).get('stream') == camera_filter
                              or (v.get('exit') or {}).get('stream') == camera_filter]
                import csv, io
                out = io.StringIO()
                writer = csv.writer(out)
                writer.writerow(['S.No', 'License Plate', 'Vehicle Type', 'Status', 'Entry Camera', 'Entry Date & Time',
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
                    writer.writerow([idx, j['plate'], normalize_vehicle_type(j.get('vehicle_type')).upper(), j['status'].upper(),
                                     entry_info.get('device', '—'), first_ts, exit_info.get('device', '—'),
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
            if path_str == '/ambulances':
                hours_q = str(query.get('hours', ['24'])[0]).lower()
                if hours_q == 'today':
                    now_sec = int(time.time())
                    hrs = max(0.25, ((now_sec + IST_OFFSET_SEC) % 86400) / 3600.0)
                else:
                    hrs = min(720, max(1, float(hours_q)))
                limit_q = min(500, max(10, int(query.get('limit', ['200'])[0])))
                return self.send_json(200, build_ambulance_list(self.store, hours=hrs, limit=limit_q))
            if path_str == '/vehicle-counts':
                now_ms = int(time.time() * 1000)
                from_q = query.get('from_ts', [None])[0]
                to_q = query.get('to_ts', [None])[0]
                hours_q = query.get('hours', ['24'])[0]
                granularity = query.get('granularity', ['auto'])[0]
                direction_q = query.get('direction', [None])[0]
                vtype_q = query.get('type', [None])[0]
                limit_q = min(20000, max(10, int(query.get('limit', ['60'])[0])))
                if from_q is not None:
                    from_ts = max(0, int(float(from_q)))
                    to_ts = int(float(to_q)) if to_q is not None else now_ms
                elif str(hours_q).lower() in ('all', '0'):
                    from_ts = 0
                    to_ts = now_ms
                elif str(hours_q).lower() == 'today':
                    now_sec = int(time.time())
                    seconds_today_in_ist = (now_sec + IST_OFFSET_SEC) % 86400
                    from_ts = (now_sec - seconds_today_in_ist) * 1000
                    to_ts = now_ms
                else:
                    hours_val = max(0.25, float(hours_q))
                    from_ts = int(now_ms - hours_val * 3600000)
                    to_ts = now_ms
                summary = self.store.vehicle_counts_summary(
                    from_ts=from_ts, to_ts=to_ts, granularity=granularity,
                    event_limit=limit_q, direction=direction_q, vehicle_type=vtype_q
                )
                summary['remaining_cameras_live'] = self.engine.remaining_cameras_live()
                return self.send_json(200, summary)
            if path_str in ('/vehicle-counts.csv', '/vehicle-counts/export'):
                now_ms = int(time.time() * 1000)
                from_q = query.get('from_ts', [None])[0]
                to_q = query.get('to_ts', [None])[0]
                hours_q = query.get('hours', ['all'])[0]
                direction = query.get('direction', [None])[0]
                vtype_q = query.get('type', [None])[0]
                if from_q is not None:
                    from_ts = max(0, int(float(from_q)))
                    to_ts = int(float(to_q)) if to_q is not None else now_ms
                elif str(hours_q).lower() in ('all', '0'):
                    from_ts = 0
                    to_ts = now_ms
                elif str(hours_q).lower() == 'today':
                    now_sec = int(time.time())
                    from_ts = (now_sec - (now_sec + IST_OFFSET_SEC) % 86400) * 1000
                    to_ts = now_ms
                else:
                    from_ts = int(now_ms - max(0.25, float(hours_q)) * 3600000)
                    to_ts = now_ms
                events = self.store.vehicle_counts_events(from_ts=from_ts, to_ts=to_ts, direction=direction, vehicle_type=vtype_q, limit=10000)
                import csv, io
                out = io.StringIO()
                writer = csv.writer(out)
                writer.writerow(['S.No', 'Date & Time (IST)', 'Direction', 'Gate / Camera', 'Gate Group', 'Matched Cameras', 'Vehicle Type', 'Plate (If Read)', 'Confidence'])
                for idx, ev in enumerate(events, 1):
                    writer.writerow([
                        idx,
                        to_ist_strftime(ev['ts'] / 1000),
                        ev['direction'].upper(),
                        ev['camera'],
                        ev.get('gate_group') or 'Default',
                        ', '.join(ev.get('merged_cameras') or [ev['camera']]),
                        ev['vehicle_type'].upper(),
                        ev['plate'] or '—',
                        f"{int(round((ev['confidence'] or 0) * 100))}%" if ev.get('confidence') is not None else '—'
                    ])
                payload = out.getvalue().encode('utf-8-sig')
                self.send_response(200)
                self.send_header('Content-Type', 'text/csv; charset=utf-8')
                self.send_header('Content-Disposition', 'attachment; filename="Medanta_Gate_Vehicle_Counts_' + to_ist_strftime(time.time(), '%Y%m%d_%H%M%S') + '.csv"')
                self.send_header('Content-Length', str(len(payload)))
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                return self.wfile.write(payload)
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
                    enabled=data.get('enabled', True),
                    gate_group=data.get('gate_group', ''),
                    count_enabled=data.get('count_enabled', True)
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
            if path_str == '/ambulances/dispatch-email':
                raw_to = data.get('to') or []
                if isinstance(raw_to, str):
                    raw_to = [x.strip() for x in raw_to.split(',')]
                to_list = [str(x).strip() for x in raw_to if x and '@' in str(x)]
                if not to_list:
                    return self.send_json(400, {'error': "At least one valid recipient email ('to') is required."})
                raw_cc = data.get('cc') or []
                if isinstance(raw_cc, str):
                    raw_cc = [x.strip() for x in raw_cc.split(',')]
                cc_list = [str(x).strip() for x in raw_cc if x and '@' in str(x)]
                subject = str(data.get('subject') or 'Medanta Ambulance Alert').strip()
                title = str(data.get('title') or '🚑 Ambulance Movement Advisory').strip()
                summary = str(data.get('summary') or '').strip()
                status_tag = str(data.get('status') or 'on_site').strip()
                metadata = dict(data.get('metadata') or {})
                note = str(data.get('note') or '').strip()
                if note:
                    metadata['Dispatcher Note'] = note

                # Attach JPEG snapshots from sightings DB (fallback to inline base64 if supplied)
                sids = [s for s in (data.get('exit_sighting_id'), data.get('entry_sighting_id'), data.get('sighting_id')) if s]
                veh_b64 = data.get('photo_b64') or data.get('body_crop_base64') or None
                plate_b64 = data.get('plate_b64') or data.get('face_crop_base64') or None
                for sid in sids:
                    if not veh_b64:
                        blob = self.store.sighting_photo(sid, 'photo')
                        if blob:
                            veh_b64 = base64.b64encode(blob).decode('ascii')
                    if not plate_b64:
                        pblob = self.store.sighting_photo(sid, 'plate')
                        if pblob:
                            plate_b64 = base64.b64encode(pblob).decode('ascii')

                custom_html = build_ambulance_email_html(
                    subject=subject,
                    title=title,
                    summary=summary,
                    metadata=metadata,
                    status_tag=status_tag,
                    has_plate_crop=bool(plate_b64),
                    has_vehicle_photo=bool(veh_b64),
                )
                dispatch_payload = {
                    'to': to_list,
                    'cc': cc_list,
                    'subject': subject,
                    'alert_type': 'general_alert',
                    'title': title,
                    'summary': summary,
                    'metadata': metadata,
                    'face_crop_base64': plate_b64,
                    'body_crop_base64': veh_b64,
                    'custom_html': custom_html,
                }
                try:
                    resp = relay_email_dispatch(dispatch_payload)
                except RuntimeError as err:
                    return self.send_json(502, {'error': str(err)[:300]})
                saved = record_ambulance_dispatch(
                    self.store,
                    ref_id=str(data.get('ref_id') or ''),
                    plate=str(data.get('plate') or ''),
                    recipients=to_list,
                    subject=subject,
                    note=note,
                    message_id=str(resp.get('message_id') or ''),
                )
                return self.send_json(200, {
                    'success': True,
                    'message_id': resp.get('message_id'),
                    'to': to_list,
                    'cc': cc_list,
                    'images_attached': {'plate_crop': bool(plate_b64), 'vehicle_photo': bool(veh_b64)},
                    'dispatch': saved,
                })
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