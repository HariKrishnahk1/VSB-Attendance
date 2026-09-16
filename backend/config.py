import os
import shutil
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# In Vercel serverless environment, the filesystem is read-only except /tmp
IS_VERCEL = bool(os.getenv("VERCEL"))
BASE_STORAGE = Path("/tmp") if IS_VERCEL else BASE_DIR

# Storage paths
UPLOAD_DIR = BASE_STORAGE / "uploads"
STUDENT_PHOTOS_DIR = UPLOAD_DIR / "student_photos"
REVIEW_CROPS_DIR = UPLOAD_DIR / "review_crops"
EXCEL_OUTPUT_DIR = UPLOAD_DIR / "excel_reports"
MODELS_DIR = BASE_STORAGE / "models" if IS_VERCEL else BASE_DIR / "models"

for directory in [UPLOAD_DIR, STUDENT_PHOTOS_DIR, REVIEW_CROPS_DIR, EXCEL_OUTPUT_DIR, MODELS_DIR]:
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

# Database & Pre-existing Files Setup
if IS_VERCEL:
    db_tmp = Path("/tmp/attendance.db")
    db_root = BASE_DIR / "attendance.db"
    if not db_tmp.exists() and db_root.exists():
        try:
            shutil.copy2(db_root, db_tmp)
            print("[Vercel Setup] Copied initial database to /tmp/attendance.db")
        except Exception as e:
            print(f"[Vercel Setup] Error copying database: {e}")

    uploads_root = BASE_DIR / "uploads"
    if uploads_root.exists():
        try:
            for item in uploads_root.rglob("*"):
                if item.is_file():
                    rel_path = item.relative_to(uploads_root)
                    target_path = UPLOAD_DIR / rel_path
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    if not target_path.exists():
                        shutil.copy2(item, target_path)
            print("[Vercel Setup] Synced pre-existing upload assets to /tmp/uploads")
        except Exception as e:
            print(f"[Vercel Setup] Error copying upload assets: {e}")

    models_root = BASE_DIR / "models"
    if models_root.exists():
        try:
            for item in models_root.glob("*.onnx"):
                target_path = MODELS_DIR / item.name
                if not target_path.exists():
                    shutil.copy2(item, target_path)
            print("[Vercel Setup] Synced ONNX vision models to /tmp/models")
        except Exception as e:
            print(f"[Vercel Setup] Error copying ONNX models: {e}")

    DATABASE_URL = "sqlite:////tmp/attendance.db"
else:
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
THRESHOLD_HIGH_CONFIDENCE = 0.45
THRESHOLD_MEDIUM_CONFIDENCE = 0.32

