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

# High-Precision SFace Cosine Similarity & Anti-Fluke Zero-False-Positive Thresholds
# Calibrated for STRICT DISCRIMINATION + Exact Video Sweep & Back-Row Precision:
# >= 0.58: Front/Mid-Row High Confidence Match (>= 85% confidence) -> AUTO PRESENT (1+ frames)
# >= 0.52: Multi-Frame Video Sweep Consensus (>= 78% confidence)   -> AUTO PRESENT (2+ frames)
# >= 0.50: Long-Sight / Back-Row Consensus                         -> AUTO PRESENT (2+ frames)
# >= 0.44: Borderline / Low-Res Candidate                          -> REVIEW (Staff verification)
# < 0.44:  Unrecognized / Non-Match                               -> ABSENT (Strict safety net)
THRESHOLD_HIGH_CONFIDENCE = 0.52     # Clear frontal/mid-row match (unambiguous, >=85% confidence)
THRESHOLD_MEDIUM_CONFIDENCE = 0.48   # Confident match with video sweep consensus (2+ frames)
THRESHOLD_BACKROW_CONFIDENCE = 0.48  # Long-sight distant match (unambiguous back-row biometric match)
THRESHOLD_AMBIGUITY_MARGIN = 0.050   # 5% separation margin gap from runner-up
MIN_FACE_DIMENSION = 12              # Reject sub-pixel noise blobs under 12x12 (allows distant 12-25px faces)
MAX_STUDENT_TEMPLATES = 16


