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
YUNET_SCORE_THRESHOLD = 0.30  # High-sensitivity for multi-distance and back-row classroom coverage
YUNET_NMS_THRESHOLD = 0.38
YUNET_TOP_K = 5000

# High-Precision SFace Cosine Similarity & Multi-Template Thresholds
# SFace Cosine Standard:
# >= 0.36: High-Confidence Front/Mid Row Match
# >= 0.26: Strong Back-Row / Distance Match with multi-template ensemble
# 0.22 - 0.25: Borderline / Review Candidate
# < 0.22: Unrecognized / Non-Face Clutter
THRESHOLD_HIGH_CONFIDENCE = 0.36
THRESHOLD_MEDIUM_CONFIDENCE = 0.26
THRESHOLD_AMBIGUITY_MARGIN = 0.015  # Margin required between top-1 and top-2 candidate
MAX_STUDENT_TEMPLATES = 24         # Maximum multi-scale + live enriched templates per student


