import os
import shutil
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Storage paths (Render container/server environment)
UPLOAD_DIR = BASE_DIR / "uploads"
STUDENT_PHOTOS_DIR = UPLOAD_DIR / "student_photos"
REVIEW_CROPS_DIR = UPLOAD_DIR / "review_crops"
EXCEL_OUTPUT_DIR = UPLOAD_DIR / "excel_reports"
MODELS_DIR = BASE_DIR / "models"

for directory in [UPLOAD_DIR, STUDENT_PHOTOS_DIR, REVIEW_CROPS_DIR, EXCEL_OUTPUT_DIR, MODELS_DIR]:
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

DATABASE_URL = f"sqlite:///{BASE_DIR}/attendance.db"

# Security & JWT
SECRET_KEY = "c1a93f8b6e2d4f5a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 1 day

# Vision & Face Recognition Thresholds
YUNET_MODEL_PATH = str(MODELS_DIR / "yunet.onnx")
SFACE_MODEL_PATH = str(MODELS_DIR / "sface.onnx")

# YuNet Face Detector parameters
YUNET_SCORE_THRESHOLD = 0.45  # Stricter detection to eliminate background clutter/false faces
YUNET_NMS_THRESHOLD = 0.35
YUNET_TOP_K = 5000

# High-Precision SFace Cosine Similarity & Multi-Template Thresholds
# Calibrated for Zero False Positives in open-set classroom environments:
# >= 0.52: Definitive High-Confidence Match (Single Frame) -> AUTO PRESENT
# >= 0.46: Multi-Frame Consensus Match (2+ Frames)         -> AUTO PRESENT
# >= 0.43: Distant / Back-Row Consensus Match (3+ Frames)  -> AUTO PRESENT
# >= 0.40: Borderline Ambiguous Match (2+ Frames)          -> REVIEW
# < 0.40:  Unrecognized / Distant Noise / Non-Match        -> ABSENT
THRESHOLD_HIGH_CONFIDENCE = 0.52   # Clear frontal/mid-row match (single-frame sufficient)
THRESHOLD_MEDIUM_CONFIDENCE = 0.46  # Confident match with multi-frame verification
THRESHOLD_BACKROW_CONFIDENCE = 0.43 # Distant match requiring 3+ consistent frames
THRESHOLD_AMBIGUITY_MARGIN = 0.035  # Minimum gap between top-1 and top-2 match (prevents ambiguous matches)
MIN_FACE_DIMENSION = 22             # Reject detections smaller than 22x22 (too small for biometric fidelity)
MAX_STUDENT_TEMPLATES = 16


