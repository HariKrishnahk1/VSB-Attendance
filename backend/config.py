import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Storage paths
UPLOAD_DIR = BASE_DIR / "uploads"
STUDENT_PHOTOS_DIR = UPLOAD_DIR / "student_photos"
REVIEW_CROPS_DIR = UPLOAD_DIR / "review_crops"
EXCEL_OUTPUT_DIR = UPLOAD_DIR / "excel_reports"
MODELS_DIR = BASE_DIR / "models"

for directory in [UPLOAD_DIR, STUDENT_PHOTOS_DIR, REVIEW_CROPS_DIR, EXCEL_OUTPUT_DIR, MODELS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Database
DATABASE_URL = f"sqlite:///{BASE_DIR}/attendance.db"

# Security & JWT
SECRET_KEY = "c1a93f8b6e2d4f5a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 1 day

# Vision & Face Recognition Thresholds
YUNET_MODEL_PATH = str(MODELS_DIR / "yunet.onnx")
SFACE_MODEL_PATH = str(MODELS_DIR / "sface.onnx")

# YuNet Face Detector parameters
YUNET_SCORE_THRESHOLD = 0.75
YUNET_NMS_THRESHOLD = 0.3
YUNET_TOP_K = 5000

# SFace Cosine Similarity Thresholds
# SFace produces 128-d vectors. Cosine similarity ranges from -1.0 to 1.0.
# Threshold typical values: >0.45 high match, >0.32 review match, <0.32 unknown
THRESHOLD_HIGH_CONFIDENCE = 0.45
THRESHOLD_MEDIUM_CONFIDENCE = 0.32
