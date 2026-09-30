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
# High-precision classroom face detection (avoids background clutter and shadow artifacts)
YUNET_SCORE_THRESHOLD = 0.28
YUNET_NMS_THRESHOLD = 0.35
YUNET_TOP_K = 5000

# High-Precision SFace Cosine Similarity & Anti-Fluke Multi-Template Thresholds
# Calibrated for STRICT ZERO FALSE POSITIVES + Exact Long-Distance Back-Row Precision:
# >= 0.65: Front/Mid-Row High Confidence Match (Unambiguous) -> AUTO PRESENT (1+ frames)
# >= 0.58: Front/Mid-Row Consensus Match                     -> AUTO PRESENT (2+ frames)
# >= 0.54: Long-Sight / Back-Row Consensus Match             -> AUTO PRESENT (requires 2+ consistent frames, margin >= 0.055)
# >= 0.44: Borderline / Low-Res Candidate                    -> REVIEW (Staff verification)
# < 0.44:  Unrecognized / Non-Match                         -> ABSENT (Strict safety net)
THRESHOLD_HIGH_CONFIDENCE = 0.65    # Clear frontal/mid-row match (unambiguous)
THRESHOLD_MEDIUM_CONFIDENCE = 0.58  # Confident match with multi-frame verification
THRESHOLD_BACKROW_CONFIDENCE = 0.54 # Long-sight distant match (requires 2+ consistent frames)
THRESHOLD_AMBIGUITY_MARGIN = 0.065  # Minimum 6.5% separation gap between top-1 and runner-up
MIN_FACE_DIMENSION = 20             # Reject sub-pixel noise blobs under 20x20
MAX_STUDENT_TEMPLATES = 16


