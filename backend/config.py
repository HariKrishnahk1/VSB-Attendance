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
YUNET_SCORE_THRESHOLD = 0.40  # Balanced sensitivity - reduces false face detections from background
YUNET_NMS_THRESHOLD = 0.38
YUNET_TOP_K = 5000

# High-Precision SFace Cosine Similarity & Multi-Template Thresholds
# SFace Cosine Calibrated Ranges (verified against SFace paper):
# >= 0.50: High-Confidence Front/Mid Row Match          -> AUTO PRESENT
# >= 0.42: Mid-row match with 2+ frame consensus        -> AUTO PRESENT
# >= 0.38: Back-row / distance match (face < 70px)      -> PRESENT if 2+ frames
# >= 0.34: Borderline / single-frame distant match      -> REVIEW
# < 0.34:  Unrecognized / Non-Face / False detection    -> ABSENT
THRESHOLD_HIGH_CONFIDENCE = 0.50   # Front/mid-row clear match
THRESHOLD_MEDIUM_CONFIDENCE = 0.38  # Back-row / distant match (requires multi-frame)
THRESHOLD_AMBIGUITY_MARGIN = 0.025  # Minimum gap between top-1 and top-2 match (prevents ambiguous matches)
MAX_STUDENT_TEMPLATES = 24          # Maximum multi-scale + live enriched templates per student


