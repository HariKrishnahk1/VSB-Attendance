import os
import zipfile
import json
import numpy as np
import cv2
import re
import urllib.request
from pathlib import Path
from backend.config import (
    YUNET_MODEL_PATH, SFACE_MODEL_PATH,
    STUDENT_PHOTOS_DIR, REVIEW_CROPS_DIR,
    YUNET_NMS_THRESHOLD
)

# Official OpenCV Pretrained ONNX Models
YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
SFACE_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"

# Optimized for 70 to 80 Students High-Density Classroom / Lecture Hall
HIGH_DENSITY_SCORE_THRESHOLD = 0.40
MAX_CLASSROOM_DETECTIONS = 5000


def _ensure_model_exists(file_path: str, url: str):
    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        print(f"[VisionEngine] Pretrained model missing at {file_path}. Auto-downloading from OpenCV Zoo ({url})...")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        try:
            urllib.request.urlretrieve(url, file_path)
            print(f"[VisionEngine] Successfully downloaded model to {file_path}!")
        except Exception as e:
            print(f"[VisionEngine] Error downloading model from {url}: {e}")


class VisionEngine:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(VisionEngine, cls).__new__(cls)
            cls._instance._init_models()
        return cls._instance

    def _init_models(self):
        print("[VisionEngine] Initializing 70-80 Student High-Density Classroom YuNet & SFace models...")
        _ensure_model_exists(YUNET_MODEL_PATH, YUNET_URL)
        _ensure_model_exists(SFACE_MODEL_PATH, SFACE_URL)

        if not os.path.exists(YUNET_MODEL_PATH) or not os.path.exists(SFACE_MODEL_PATH):
            raise FileNotFoundError(
                f"Vision models missing at {YUNET_MODEL_PATH} or {SFACE_MODEL_PATH}."
            )

        self.detector = cv2.FaceDetectorYN.create(
            YUNET_MODEL_PATH,
            "",
            (640, 640),
            HIGH_DENSITY_SCORE_THRESHOLD,
            YUNET_NMS_THRESHOLD,
            MAX_CLASSROOM_DETECTIONS
        )
        self.recognizer = cv2.FaceRecognizerSF.create(SFACE_MODEL_PATH, "")
        print("[VisionEngine] 70-80 Student High-Density Vision Engine Ready!")

    def detect_faces(self, img_bgr):
        """
        Detects faces across a dense 70-80 student classroom layout.
        Uses 3x3 High-Density Grid Tiling + Full Frame Pass.
        """
        h, w, _ = img_bgr.shape
        all_detected_faces = []

        # 1. Full Frame High-Res Pass
        self.detector.setInputSize((w, h))
        _, faces_main = self.detector.detect(img_bgr)
        if faces_main is not None:
            for f in faces_main:
                all_detected_faces.append(f)

        # 2. 3x3 Grid Tiling Pass (scans all 70-80 student seating positions in high detail)
        qw, qh = int(w * 0.4), int(h * 0.4)
        overlap_x = int(w * 0.3)
        overlap_y = int(h * 0.3)

        tiles = [
            # Row 1 (Back Row Students - 25 to 30 ft)
            (0, 0, qw, qh), (overlap_x, 0, qw, qh), (w - qw, 0, qw, qh),
            # Row 2 (Middle Row Students - 15 to 20 ft)
            (0, overlap_y, qw, qh), (overlap_x, overlap_y, qw, qh), (w - qw, overlap_y, qw, qh),
            # Row 3 (Front Row Students - 5 to 10 ft)
            (0, h - qh, qw, qh), (overlap_x, h - qh, qw, qh), (w - qw, h - qh, qw, qh)
        ]

        for tx, ty, tw, th in tiles:
            tile_crop = img_bgr[ty:ty+th, tx:tx+tw]
            if tile_crop.size == 0:
                continue

            # Upscale 1.6x for distant small face clarity
            scaled_tile = cv2.resize(tile_crop, (int(tw * 1.6), int(th * 1.6)), interpolation=cv2.INTER_CUBIC)
            sth, stw, _ = scaled_tile.shape
            self.detector.setInputSize((stw, sth))

            _, tile_faces = self.detector.detect(scaled_tile)
            if tile_faces is not None:
                scale_back = 1.0 / 1.6
                for f in tile_faces:
                    f_mapped = f.copy()
                    f_mapped[0] = tx + f[0] * scale_back
                    f_mapped[1] = ty + f[1] * scale_back
                    f_mapped[2] = f[2] * scale_back
                    f_mapped[3] = f[3] * scale_back
                    for l_idx in range(4, 14, 2):
                        f_mapped[l_idx] = tx + f[l_idx] * scale_back
                        f_mapped[l_idx + 1] = ty + f[l_idx + 1] * scale_back
                    all_detected_faces.append(f_mapped)

        if not all_detected_faces:
            return []

        # Deduplicate overlapping bounding boxes (NMS IoU)
        return self._suppress_duplicate_faces(all_detected_faces)

    def _suppress_duplicate_faces(self, face_list, iou_threshold=0.35):
        if not face_list:
            return []

        sorted_faces = sorted(face_list, key=lambda f: f[14], reverse=True)
        keep = []

        while sorted_faces:
            best = sorted_faces.pop(0)
            keep.append(best)

            remaining = []
            for other in sorted_faces:
                iou = self._calc_iou(best[:4], other[:4])
                if iou < iou_threshold:
                    remaining.append(other)
            sorted_faces = remaining

        return keep

    def _calc_iou(self, box1, box2):
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)

        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        union_area = (w1 * h1) + (w2 * h2) - inter_area
        return inter_area / union_area if union_area > 0 else 0.0

    def extract_embedding(self, img_bgr, face_data):
        w, h = face_data[2], face_data[3]

        if w < 50 or h < 50:
            box = face_data[:4].astype(int)
            x, y, bw, bh = box
            img_h, img_w, _ = img_bgr.shape
            x, y = max(0, x), max(0, y)
            
            crop = img_bgr[y:min(y+bh, img_h), x:min(x+bw, img_w)]
            if crop.size > 0:
                aligned_face = cv2.resize(crop, (112, 112), interpolation=cv2.INTER_CUBIC)
            else:
                aligned_face = self.recognizer.alignCrop(img_bgr, face_data)
        else:
            aligned_face = self.recognizer.alignCrop(img_bgr, face_data)

        feature = self.recognizer.feature(aligned_face)
        return feature.flatten()

    def compare_embeddings_batch(self, detected_vec, ref_matrix):
        """
        Fast Vectorized Cosine Similarity for 70-80 students in < 1ms.
        ref_matrix: (N_students, 128) numpy array
        detected_vec: (128,) numpy array
        Returns: numpy array of cosine similarity scores for all N_students.
        """
        if ref_matrix.size == 0:
            return np.array([])
        
        # Unit normalize
        norm_det = np.linalg.norm(detected_vec)
        if norm_det == 0:
            norm_det = 1.0
        det_unit = detected_vec / norm_det

        norm_refs = np.linalg.norm(ref_matrix, axis=1, keepdims=True)
        norm_refs[norm_refs == 0] = 1.0
        refs_unit = ref_matrix / norm_refs

        scores = np.dot(refs_unit, det_unit)
        return scores

    def compare_embeddings(self, emb1, emb2):
        f1 = np.array(emb1, dtype=np.float32).reshape(1, -1)
        f2 = np.array(emb2, dtype=np.float32).reshape(1, -1)
        score = self.recognizer.match(f1, f2, cv2.FaceRecognizerSF_FR_COSINE)
        return float(score)

    def check_blurriness(self, img_bgr):
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def process_single_image(self, img_bytes_or_path):
        if isinstance(img_bytes_or_path, (str, Path)):
            img = cv2.imread(str(img_bytes_or_path))
        else:
            np_arr = np.frombuffer(img_bytes_or_path, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            return {"status": "INVALID_IMAGE", "message": "Failed to decode image file."}

        blur_score = self.check_blurriness(img)
        if blur_score < 20.0:
            return {"status": "POOR_QUALITY", "message": f"Image is blurry (blur score: {blur_score:.1f})"}

        faces = self.detect_faces(img)
        if len(faces) == 0:
            return {"status": "NO_FACE", "message": "No face detected in image."}

        face = max(faces, key=lambda f: f[2] * f[3])
        embedding = self.extract_embedding(img, face)

        return {
            "status": "SUCCESS",
            "face_box": face[:4].tolist(),
            "embedding": embedding.tolist(),
            "quality_score": blur_score,
            "img_bgr": img,
            "face_data": face
        }


def get_vision_engine():
    return VisionEngine()


def process_zip_dataset(zip_file_path: str, class_id: int, db_session):
    from backend.models import Student, StudentFaceEmbedding

    vision = get_vision_engine()
    extract_temp_dir = STUDENT_PHOTOS_DIR / f"temp_zip_{class_id}"
    extract_temp_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "total_students": 0,
        "successfully_processed": 0,
        "invalid_images": 0,
        "multiple_face_images": 0,
        "no_face_images": 0,
        "duplicate_ids": 0,
        "poor_quality_images": 0,
        "details": []
    }

    seen_ids = set()

    try:
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            for member in zip_ref.namelist():
                filename = os.path.basename(member)
                if not filename or member.startswith('__MACOSX') or filename.startswith('.'):
                    continue
                
                ext = Path(filename).suffix.lower()
                if ext not in ['.jpg', '.jpeg', '.png']:
                    continue

                summary["total_students"] += 1
                source_stream = zip_ref.read(member)
                
                base_name = Path(filename).stem
                parts = re.split(r'[_ -]+', base_name)
                student_reg = parts[0]
                student_name = " ".join(parts[1:]) if len(parts) > 1 else f"Student {student_reg}"

                if student_reg in seen_ids:
                    summary["duplicate_ids"] += 1
                    summary["details"].append({
                        "filename": filename,
                        "student_id": student_reg,
                        "status": "DUPLICATE_ID",
                        "message": f"Duplicate Student ID {student_reg} found in ZIP."
                    })
                    continue
                seen_ids.add(student_reg)

                res = vision.process_single_image(source_stream)
                status = res["status"]

                if status == "SUCCESS":
                    saved_photo_name = f"{class_id}_{student_reg}.jpg"
                    saved_photo_path = STUDENT_PHOTOS_DIR / saved_photo_name
                    cv2.imwrite(str(saved_photo_path), res["img_bgr"])

                    student = db_session.query(Student).filter(
                        Student.student_id == student_reg
                    ).first()

                    if not student:
                        student = Student(
                            student_id=student_reg,
                            name=student_name,
                            class_id=class_id,
                            photo_path=f"/static/uploads/student_photos/{saved_photo_name}"
                        )
                        db_session.add(student)
                        db_session.flush()
                    else:
                        student.class_id = class_id
                        student.name = student_name
                        student.photo_path = f"/static/uploads/student_photos/{saved_photo_name}"

                    db_session.query(StudentFaceEmbedding).filter(
                        StudentFaceEmbedding.student_id == student.id
                    ).delete()

                    embedding_record = StudentFaceEmbedding(
                        student_id=student.id,
                        embedding_data=json.dumps(res["embedding"]),
                        quality_score=res["quality_score"]
                    )
                    db_session.add(embedding_record)
                    summary["successfully_processed"] += 1
                    summary["details"].append({
                        "filename": filename,
                        "student_id": student_reg,
                        "name": student_name,
                        "status": "SUCCESS",
                        "message": "Processed successfully"
                    })
                else:
                    if status == "NO_FACE":
                        summary["no_face_images"] += 1
                    elif status == "MULTIPLE_FACES":
                        summary["multiple_face_images"] += 1
                    elif status == "POOR_QUALITY":
                        summary["poor_quality_images"] += 1
                    else:
                        summary["invalid_images"] += 1

                    summary["details"].append({
                        "filename": filename,
                        "student_id": student_reg,
                        "status": status,
                        "message": res.get("message", "Processing failed.")
                    })

        db_session.commit()
    except Exception as e:
        db_session.rollback()
        raise e

    return summary
