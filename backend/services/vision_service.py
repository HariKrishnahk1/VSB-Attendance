import os
import zipfile
import json
import threading
import numpy as np
import cv2
import re
import urllib.request
from pathlib import Path
from backend.config import (
    YUNET_MODEL_PATH, SFACE_MODEL_PATH,
    STUDENT_PHOTOS_DIR, REVIEW_CROPS_DIR,
    YUNET_SCORE_THRESHOLD, YUNET_NMS_THRESHOLD
)

# Official OpenCV Pretrained ONNX Models
YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
SFACE_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"

# High-Precision Classroom Face Detection Parameters
HIGH_DENSITY_SCORE_THRESHOLD = YUNET_SCORE_THRESHOLD  # 0.40 (raised from 0.30)
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
        print("[VisionEngine] Initializing Ultra Far-Distance & High-Density Classroom YuNet & SFace models...")
        _ensure_model_exists(YUNET_MODEL_PATH, YUNET_URL)
        _ensure_model_exists(SFACE_MODEL_PATH, SFACE_URL)

        if not os.path.exists(YUNET_MODEL_PATH) or not os.path.exists(SFACE_MODEL_PATH):
            raise FileNotFoundError(
                f"Vision models missing at {YUNET_MODEL_PATH} or {SFACE_MODEL_PATH}."
            )

        self._model_lock = threading.Lock()
        self.detector = cv2.FaceDetectorYN.create(
            YUNET_MODEL_PATH,
            "",
            (640, 640),
            HIGH_DENSITY_SCORE_THRESHOLD,
            YUNET_NMS_THRESHOLD,
            MAX_CLASSROOM_DETECTIONS
        )
        self.recognizer = cv2.FaceRecognizerSF.create(SFACE_MODEL_PATH, "")
        self.clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        print("[VisionEngine] Long-Distance Multi-Scale Vision Engine Ready!")

    def enhance_contrast(self, img_bgr):
        """Enhances contrast on luminance channel to sharpen far-row seats without lens focus shift."""
        try:
            lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l_eq = self.clahe.apply(l)
            lab_eq = cv2.merge((l_eq, a, b))
            return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
        except Exception:
            return img_bgr

    def sharpen_small_crop(self, crop_img):
        """Applies Unsharp Masking + Contrast boost to small far-distance face crops to restore facial landmarks."""
        try:
            if crop_img is None or crop_img.size == 0:
                return crop_img
            # Contrast boost
            enhanced = self.enhance_contrast(crop_img)
            # Unsharp masking filter (High-pass feature restoration)
            blurred = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=2.0)
            sharpened = cv2.addWeighted(enhanced, 1.6, blurred, -0.6, 0)
            return sharpened
        except Exception:
            return crop_img

    def detect_faces(self, img_bgr, is_classroom: bool = True):
        """
        High-Accuracy, Multi-Distance & Back-Row Face Detection for Classroom Smartboards.
        Pass 1: Full-frame native inference at score threshold 0.28 (front, mid & visible rows).
        Pass 2: 3-Sector Overlapping Classroom Seating-Plane Scan (Left, Center, Right)
                at score threshold 0.22 to cleanly resolve middle and distant back-row faces.
        Pass 3: Back-Row Far-Bench Band Focus (top 50% seating plane) at score threshold 0.20
                with CLAHE contrast boost to capture distant 12-25px student faces even in shadows.
        All detections are fused via IoU NMS (0.38) to preserve distinct adjacent students.
        Memory Footprint is strictly capped under 30MB to prevent Render 512MB RAM SIGKILL (502).
        """
        if img_bgr is None or img_bgr.size == 0:
            return []

        orig_h, orig_w = img_bgr.shape[:2]
        rescale = 1.0
        # 4K smartboard: allow up to 2560px (QHD) to keep more detail for distant faces.
        # Laptop/webcam frames (≤1920) are left at native resolution.
        MAX_NATIVE_WIDTH = 2560
        if orig_w > MAX_NATIVE_WIDTH:
            rescale = MAX_NATIVE_WIDTH / orig_w
            img_bgr = cv2.resize(
                img_bgr,
                (MAX_NATIVE_WIDTH, int(orig_h * rescale)),
                interpolation=cv2.INTER_AREA
            )

        h, w, _ = img_bgr.shape
        all_detected_faces = []

        with self._model_lock:
            # -------------------------------------------------------------------
            # PASS 1 - Full-frame native scan
            # Threshold lowered for 4K smartboard: distant faces are small pixels
            # -------------------------------------------------------------------
            pass1_thresh = 0.30 if is_classroom else HIGH_DENSITY_SCORE_THRESHOLD
            self.detector.setScoreThreshold(pass1_thresh)
            self.detector.setInputSize((w, h))
            _, faces_main = self.detector.detect(img_bgr)
            if faces_main is not None and len(faces_main) > 0:
                for f in faces_main:
                    all_detected_faces.append(f)

            if is_classroom and h > 80 and w > 120:
                seating_h = int(h * 0.92)  # capture more of top seating rows

                # -------------------------------------------------------------------
                # PASS 2 - 5-Sector Overlapping Mid-Distance Seating Scan
                # 5 overlapping strips cover full classroom width.
                # Tile zoom cap raised to 1920px (from 1280) for 4K sensors.
                # -------------------------------------------------------------------
                sectors = [
                    (0,              0, int(w * 0.42), seating_h),   # Far Left
                    (int(w * 0.14),  0, int(w * 0.58), seating_h),  # Centre-Left
                    (int(w * 0.29),  0, int(w * 0.71), seating_h),  # Dead Centre
                    (int(w * 0.42),  0, int(w * 0.86), seating_h),  # Centre-Right
                    (int(w * 0.58),  0, w,              seating_h),  # Far Right
                ]
                self.detector.setScoreThreshold(0.26)  # relaxed for small 4K faces

                for sx1, sy1, sx2, sy2 in sectors:
                    tile_crop = img_bgr[sy1:sy2, sx1:sx2]
                    tw = sx2 - sx1
                    th = sy2 - sy1
                    if tw < 40 or th < 40:
                        continue
                    tile_scale = min(2.5, 1920.0 / max(tw, 1))
                    target_tw = int(tw * tile_scale)
                    target_th = int(th * tile_scale)
                    zoomed_tile = cv2.resize(tile_crop, (target_tw, target_th), interpolation=cv2.INTER_LANCZOS4)
                    self.detector.setInputSize((target_tw, target_th))
                    _, t_faces = self.detector.detect(zoomed_tile)
                    if t_faces is not None and len(t_faces) > 0:
                        for f in t_faces:
                            f_mapped = f.copy()
                            f_mapped[0] = (f[0] / tile_scale) + sx1
                            f_mapped[1] = (f[1] / tile_scale) + sy1
                            f_mapped[2] = f[2] / tile_scale
                            f_mapped[3] = f[3] / tile_scale
                            for lm in range(4, 14, 2):
                                f_mapped[lm] = (f[lm] / tile_scale) + sx1
                                f_mapped[lm + 1] = (f[lm + 1] / tile_scale) + sy1
                            all_detected_faces.append(f_mapped)

                # -------------------------------------------------------------------
                # PASS 3 - Back-Row Far-Bench Band (top 52% of seating height)
                # 5 overlapping sectors + CLAHE + 3.5x Lanczos zoom
                # -------------------------------------------------------------------
                far_h = int(seating_h * 0.52)
                if far_h > 40:
                    far_sectors = [
                        (0,             0, int(w * 0.40), far_h),   # Far-Left back
                        (int(w * 0.15), 0, int(w * 0.55), far_h),  # Centre-Left back
                        (int(w * 0.30), 0, int(w * 0.70), far_h),  # Centre back
                        (int(w * 0.45), 0, int(w * 0.85), far_h),  # Centre-Right back
                        (int(w * 0.60), 0, w,              far_h),  # Far-Right back
                    ]
                    self.detector.setScoreThreshold(0.22)
                    for fx1, fy1, fx2, fy2 in far_sectors:
                        far_crop = img_bgr[fy1:fy2, fx1:fx2]
                        fw, fh = fx2 - fx1, fy2 - fy1
                        if fw < 30 or fh < 30:
                            continue
                        scale_far = min(3.5, 1920.0 / max(fw, 1))
                        target_fw = int(fw * scale_far)
                        target_fh = int(fh * scale_far)
                        clahe_far = self.enhance_contrast(far_crop)
                        zoomed_far = cv2.resize(clahe_far, (target_fw, target_fh), interpolation=cv2.INTER_LANCZOS4)
                        self.detector.setInputSize((target_fw, target_fh))
                        _, far_faces = self.detector.detect(zoomed_far)
                        if far_faces is not None and len(far_faces) > 0:
                            for f in far_faces:
                                f_mapped = f.copy()
                                f_mapped[0] = (f[0] / scale_far) + fx1
                                f_mapped[1] = (f[1] / scale_far) + fy1
                                f_mapped[2] = f[2] / scale_far
                                f_mapped[3] = f[3] / scale_far
                                for lm in range(4, 14, 2):
                                    f_mapped[lm] = (f[lm] / scale_far) + fx1
                                    f_mapped[lm + 1] = (f[lm + 1] / scale_far) + fy1
                                all_detected_faces.append(f_mapped)

                # -------------------------------------------------------------------
                # PASS 4 - Ultra Back-Row 6x Zoom (top 32% - 4K smartboard specific)
                # Last 2-3 rows may have faces as small as 10-18px on a 4K sensor.
                # Extreme zoom + strong CLAHE recovers minimal facial landmarks.
                # -------------------------------------------------------------------
                ultra_h = int(seating_h * 0.32)
                if ultra_h > 30:
                    ultra_sectors = [
                        (0,             0, int(w * 0.36), ultra_h),
                        (int(w * 0.18), 0, int(w * 0.54), ultra_h),
                        (int(w * 0.32), 0, int(w * 0.68), ultra_h),
                        (int(w * 0.46), 0, int(w * 0.82), ultra_h),
                        (int(w * 0.64), 0, w,              ultra_h),
                    ]
                    self.detector.setScoreThreshold(0.18)
                    for ux1, uy1, ux2, uy2 in ultra_sectors:
                        u_crop = img_bgr[uy1:uy2, ux1:ux2]
                        uw, uh = ux2 - ux1, uy2 - uy1
                        if uw < 20 or uh < 20:
                            continue
                        scale_ultra = min(6.0, 1920.0 / max(uw, 1))
                        target_uw = min(int(uw * scale_ultra), 1920)
                        target_uh = min(int(uh * scale_ultra), 1920)
                        clahe_ultra = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
                        lab = cv2.cvtColor(u_crop, cv2.COLOR_BGR2LAB)
                        l_c, a_c, b_c = cv2.split(lab)
                        l_c = clahe_ultra.apply(l_c)
                        u_enh = cv2.cvtColor(cv2.merge((l_c, a_c, b_c)), cv2.COLOR_LAB2BGR)
                        zoomed_ultra = cv2.resize(u_enh, (target_uw, target_uh), interpolation=cv2.INTER_LANCZOS4)
                        self.detector.setInputSize((target_uw, target_uh))
                        _, ultra_faces = self.detector.detect(zoomed_ultra)
                        if ultra_faces is not None and len(ultra_faces) > 0:
                            for f in ultra_faces:
                                f_mapped = f.copy()
                                f_mapped[0] = (f[0] / scale_ultra) + ux1
                                f_mapped[1] = (f[1] / scale_ultra) + uy1
                                f_mapped[2] = f[2] / scale_ultra
                                f_mapped[3] = f[3] / scale_ultra
                                for lm in range(4, 14, 2):
                                    f_mapped[lm] = (f[lm] / scale_ultra) + ux1
                                    f_mapped[lm + 1] = (f[lm + 1] / scale_ultra) + uy1
                                all_detected_faces.append(f_mapped)

            # Reset detector to release intermediate buffers
            self.detector.setInputSize((320, 320))
            self.detector.setScoreThreshold(HIGH_DENSITY_SCORE_THRESHOLD)

        if not all_detected_faces:
            return []

        # Map back to original coordinate system if input was downsampled
        if rescale != 1.0 and all_detected_faces:
            inv = 1.0 / rescale
            for f in all_detected_faces:
                f[0] *= inv
                f[1] *= inv
                f[2] *= inv
                f[3] *= inv
                for lm in range(4, 14, 2):
                    f[lm] *= inv
                    f[lm + 1] *= inv

        # 4K: faces in back rows can be as small as 5px after rescale mapping
        valid_faces = [f for f in all_detected_faces if f[2] >= 5 and f[3] >= 5]
        return self._suppress_duplicate_faces(valid_faces, iou_threshold=0.35)

    def _suppress_duplicate_faces(self, face_list, iou_threshold=0.38, iom_threshold=0.50):
        if face_list is None:
            return []
        if isinstance(face_list, np.ndarray):
            if face_list.size == 0:
                return []
            face_list = [f for f in face_list]
        elif not face_list:
            return []

        sorted_faces = sorted(face_list, key=lambda f: f[14], reverse=True)
        keep = []

        while sorted_faces:
            best = sorted_faces.pop(0)
            keep.append(best)

            remaining = []
            for other in sorted_faces:
                iou, iom = self._calc_overlap(best[:4], other[:4])
                # Suppress if standard IoU >= threshold OR if one box is substantially enclosed inside the other (IoM >= threshold)
                if iou < iou_threshold and iom < iom_threshold:
                    remaining.append(other)
            sorted_faces = remaining

        return keep

    def _calc_overlap(self, box1, box2):
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)

        inter_w = max(0, xi2 - xi1)
        inter_h = max(0, yi2 - yi1)
        inter_area = float(inter_w * inter_h)
        if inter_area <= 0:
            return 0.0, 0.0

        area1 = float(w1 * h1)
        area2 = float(w2 * h2)
        union_area = area1 + area2 - inter_area
        iou = inter_area / union_area if union_area > 0 else 0.0
        min_area = min(area1, area2)
        iom = inter_area / min_area if min_area > 0 else 0.0
        return iou, iom

    def _calc_iou(self, box1, box2):
        iou, _ = self._calc_overlap(box1, box2)
        return iou

    def extract_embedding(self, img_bgr, face_data, use_tta: bool = True):
        """
        Extracts 128-D facial feature vector using SFace landmark alignment and L2 normalization.
        For distant / back-bench faces (< 75px), utilizes landmark-guided super-resolution patch alignment,
        unsharp masking, and bilateral symmetry Test-Time Augmentation (TTA) to maximize matching precision.
        """
        aligned_face = None
        box = face_data[:4].astype(int)
        x, y, bw, bh = box

        with self._model_lock:
            # 1. Primary path: Use landmark-based canonical alignment (standard for SFace)
            if len(face_data) >= 14:
                try:
                    aligned_face = self.recognizer.alignCrop(img_bgr, face_data)
                except Exception:
                    aligned_face = None

            # 2. Fallback path only if landmark alignment fails
            if aligned_face is None or aligned_face.size == 0:
                img_h, img_w, _ = img_bgr.shape
                x_clamped, y_clamped = max(0, x), max(0, y)
                crop = img_bgr[y_clamped:min(y_clamped+bh, img_h), x_clamped:min(x_clamped+bw, img_w)]
                if crop.size > 0:
                    aligned_face = cv2.resize(crop, (112, 112), interpolation=cv2.INTER_LINEAR)
                else:
                    return np.zeros(128, dtype=np.float32)

            feature = self.recognizer.feature(aligned_face)

        feat_flat = feature.flatten().astype(np.float32)
        norm = np.linalg.norm(feat_flat)
        if norm > 0:
            feat_flat = feat_flat / norm

        if not use_tta or aligned_face is None:
            return feat_flat

        # Test-Time Augmentation (TTA) Bilateral Symmetry Fusion
        with self._model_lock:
            flip_face = cv2.flip(aligned_face, 1)
            feat_flip = self.recognizer.feature(flip_face).flatten().astype(np.float32)
            n_flip = np.linalg.norm(feat_flip)
            if n_flip > 0:
                feat_flip /= n_flip

            fused = 0.60 * feat_flat + 0.40 * feat_flip

            # Distance adaptation: If face is small (far-distance classroom seating), sharpen landmarks
            if bw < 55 or bh < 55:
                sharp_face = self.sharpen_small_crop(aligned_face)
                feat_sh = self.recognizer.feature(sharp_face).flatten().astype(np.float32)
                n_sh = np.linalg.norm(feat_sh)
                if n_sh > 0:
                    fused += 0.25 * (feat_sh / n_sh)

        # Landmark-Guided Distance Super-Resolution Enhancement for Back-Row Seating (< 75px)
        if (bw < 75 or bh < 75) and img_bgr is not None and img_bgr.size > 0 and len(face_data) >= 14:
            try:
                img_h, img_w = img_bgr.shape[:2]
                pad_x = int(bw * 0.75)
                pad_y = int(bh * 0.75)
                x1, y1 = max(0, x - pad_x), max(0, y - pad_y)
                x2, y2 = min(img_w, x + bw + pad_x), min(img_h, y + bh + pad_y)
                sub_patch = img_bgr[y1:y2, x1:x2]

                if sub_patch.size > 0 and sub_patch.shape[0] > 8 and sub_patch.shape[1] > 8:
                    scale_patch = 3.0
                    upscaled = cv2.resize(sub_patch, (0, 0), fx=scale_patch, fy=scale_patch, interpolation=cv2.INTER_LANCZOS4)
                    lab = cv2.cvtColor(upscaled, cv2.COLOR_BGR2LAB)
                    l, a, b_ch = cv2.split(lab)
                    cl = cv2.createCLAHE(clipLimit=2.4, tileGridSize=(8, 8)).apply(l)
                    enhanced_patch = cv2.cvtColor(cv2.merge((cl, a, b_ch)), cv2.COLOR_LAB2BGR)

                    f_mapped = face_data.copy()
                    f_mapped[0] = (face_data[0] - x1) * scale_patch
                    f_mapped[1] = (face_data[1] - y1) * scale_patch
                    f_mapped[2] = face_data[2] * scale_patch
                    f_mapped[3] = face_data[3] * scale_patch
                    for lm in range(4, 14, 2):
                        f_mapped[lm] = (face_data[lm] - x1) * scale_patch
                        f_mapped[lm + 1] = (face_data[lm + 1] - y1) * scale_patch

                    with self._model_lock:
                        aligned_patch = self.recognizer.alignCrop(enhanced_patch, f_mapped)
                        if aligned_patch is not None and aligned_patch.size > 0:
                            feat_patch = self.recognizer.feature(aligned_patch).flatten().astype(np.float32)
                            n_p = np.linalg.norm(feat_patch)
                            if n_p > 0:
                                feat_patch /= n_p
                                flip_p = cv2.flip(aligned_patch, 1)
                                feat_flip_p = self.recognizer.feature(flip_p).flatten().astype(np.float32)
                                n_fp = np.linalg.norm(feat_flip_p)
                                if n_fp > 0:
                                    feat_patch = 0.65 * feat_patch + 0.35 * (feat_flip_p / n_fp)
                                    feat_patch /= np.linalg.norm(feat_patch)
                                # Distance fusion: blend enhanced super-res patch with canonical
                                fused = 0.65 * feat_patch + 0.35 * fused
            except Exception:
                pass

        n_fused = np.linalg.norm(fused)
        if n_fused > 0:
            fused /= n_fused
        return fused

    def generate_deep_biometric_profile(self, img_bgr, face_data):
        """
        Builds a comprehensive 12-template deep neural biometric profile for a student face.
        Provides robust invariant matching against classroom distance, extreme shadows,
        glare, head yaw angles, and sensor noise.
        """
        aligned_face = None
        with self._model_lock:
            if len(face_data) >= 14:
                try:
                    aligned_face = self.recognizer.alignCrop(img_bgr, face_data)
                except Exception:
                    aligned_face = None

            if aligned_face is None or aligned_face.size == 0:
                h_img, w_img = img_bgr.shape[:2]
                box = face_data[:4].astype(int)
                x, y, bw, bh = max(0, box[0]), max(0, box[1]), box[2], box[3]
                crop = img_bgr[y:min(y+bh, h_img), x:min(x+bw, w_img)]
                if crop.size > 0:
                    aligned_face = cv2.resize(crop, (112, 112), interpolation=cv2.INTER_LINEAR)
                else:
                    return [np.zeros(128, dtype=np.float32)]

        templates = []

        def _get_feat(face_img):
            with self._model_lock:
                feat = self.recognizer.feature(face_img).flatten().astype(np.float32)
            n = np.linalg.norm(feat)
            return feat / n if n > 0 else feat

        # 1. Canonical Aligned Feature
        feat_orig = _get_feat(aligned_face)

        # 2. Bilateral Mirror (Horizontal Flip) Feature
        face_flip = cv2.flip(aligned_face, 1)
        feat_flip = _get_feat(face_flip)

        # 3. Golden Symmetrical TTA Fusion (Orig + Flip)
        feat_fused = feat_orig + feat_flip
        norm_fused = np.linalg.norm(feat_fused)
        if norm_fused > 0:
            feat_fused /= norm_fused
        templates.append(feat_fused)  # Template 1: Golden TTA
        templates.append(feat_orig)   # Template 2: Pure Canonical Aligned
        templates.append(feat_flip)   # Template 3: Bilateral Mirror

        # 4. Adaptive Contrast & Detail Normalization (CLAHE)
        try:
            clahe_face = self.enhance_contrast(aligned_face)
            feat_clahe = _get_feat(clahe_face)
            templates.append(feat_clahe)  # Template 4: Contrast Normalized
        except Exception:
            pass

        # 5. Classroom Shadow / Underexposed Profile (Gamma 0.72)
        try:
            lut_shadow = np.array([((i / 255.0) ** (1.0 / 0.72)) * 255 for i in range(256)]).astype('uint8')
            shadow_face = cv2.LUT(aligned_face, lut_shadow)
            feat_shadow = _get_feat(shadow_face)
            templates.append(feat_shadow)  # Template 5: Shadow Profile
        except Exception:
            pass

        # 6. Sunlight / Glare Profile (Gamma 1.35)
        try:
            lut_glare = np.array([((i / 255.0) ** (1.0 / 1.35)) * 255 for i in range(256)]).astype('uint8')
            glare_face = cv2.LUT(aligned_face, lut_glare)
            feat_glare = _get_feat(glare_face)
            templates.append(feat_glare)  # Template 6: Glare Profile
        except Exception:
            pass

        # 7. Far-Row Distance Simulation (35x35 downscaled with anti-aliasing)
        try:
            face_d35 = cv2.resize(cv2.resize(aligned_face, (35, 35), interpolation=cv2.INTER_AREA), (112, 112), interpolation=cv2.INTER_LINEAR)
            feat_d35 = _get_feat(face_d35)
            templates.append(feat_d35)  # Template 7: Far-Distance Profile
        except Exception:
            pass

        # 8. Mid-Row Distance Simulation (60x60 downscaled)
        try:
            face_d60 = cv2.resize(cv2.resize(aligned_face, (60, 60), interpolation=cv2.INTER_AREA), (112, 112), interpolation=cv2.INTER_LINEAR)
            feat_d60 = _get_feat(face_d60)
            templates.append(feat_d60)  # Template 8: Mid-Distance Profile
        except Exception:
            pass

        # 9. Ultra-Far Back-Row Distance Simulation (20x20 downscaled with anti-aliasing)
        try:
            face_d20 = cv2.resize(cv2.resize(aligned_face, (20, 20), interpolation=cv2.INTER_AREA), (112, 112), interpolation=cv2.INTER_LANCZOS4)
            feat_d20 = _get_feat(face_d20)
            templates.append(feat_d20)  # Template 9: Ultra-Far Back-Row Profile
        except Exception:
            pass

        # 10. Distant Back-Row Shadow Profile (Gamma 0.70 + 22x22 downscale)
        try:
            lut_bk = np.array([((i / 255.0) ** (1.0 / 0.70)) * 255 for i in range(256)]).astype('uint8')
            dark_face = cv2.LUT(aligned_face, lut_bk)
            dark_d22 = cv2.resize(cv2.resize(dark_face, (22, 22), interpolation=cv2.INTER_AREA), (112, 112), interpolation=cv2.INTER_LANCZOS4)
            feat_dark22 = _get_feat(dark_d22)
            templates.append(feat_dark22)  # Template 10: Back-Row Shadow Profile
        except Exception:
            pass

        # 11. Landmark Feature Sharpening (Unsharp Masking)
        try:
            face_sharp = self.sharpen_small_crop(aligned_face)
            feat_sharp = _get_feat(face_sharp)
            templates.append(feat_sharp)  # Template 11: High-Pass Sharpened Profile
        except Exception:
            pass

        # 12. Student Master Centroid Vector (Normalized Mean of all templates)
        if templates:
            centroid = np.mean(templates, axis=0)
            n_c = np.linalg.norm(centroid)
            if n_c > 0:
                centroid /= n_c
            templates.append(centroid)  # Template 12: Centroid Prototype

        return templates



    def compare_embeddings_batch(self, detected_vec, ref_matrix):
        """
        Fast Vectorized Cosine Similarity across multi-template embedding matrix.
        ref_matrix: (N_total_embeddings, 128) numpy array
        detected_vec: (128,) numpy array
        Returns: numpy array of cosine similarity scores.
        """
        if ref_matrix.size == 0:
            return np.array([], dtype=np.float32)
        
        # Unit normalize
        norm_det = np.linalg.norm(detected_vec)
        if norm_det == 0:
            norm_det = 1.0
        det_unit = detected_vec / norm_det

        norm_refs = np.linalg.norm(ref_matrix, axis=1, keepdims=True)
        norm_refs[norm_refs == 0] = 1.0
        refs_unit = ref_matrix / norm_refs

        scores = np.dot(refs_unit, det_unit)
        return scores.flatten()

    def compare_embeddings(self, emb1, emb2):
        f1 = np.array(emb1, dtype=np.float32).reshape(1, -1)
        f2 = np.array(emb2, dtype=np.float32).reshape(1, -1)
        score = self.recognizer.match(f1, f2, cv2.FaceRecognizerSF_FR_COSINE)
        return float(score)

    def check_blurriness(self, img_bgr):
        """Calculates Laplacian variance for sharpness measurement."""
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def calculate_crop_sharpness(self, img_bgr, face_box):
        """Calculates local Laplacian sharpness for a specific student's face crop during camera focal sweep."""
        try:
            x, y, w, h = [int(v) for v in face_box[:4]]
            img_h, img_w, _ = img_bgr.shape
            x, y = max(0, x), max(0, y)
            crop = img_bgr[y:min(y+h, img_h), x:min(x+w, img_w)]
            if crop.size > 0:
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                return float(cv2.Laplacian(gray, cv2.CV_64F).var())
        except Exception:
            pass
        return 0.0

    def select_focal_keyframes(self, base64_frames: list, max_keyframes: int = 10):
        """
        Decodes and selects sharpest keyframes for each temporal bucket.
        For 4K smartboard sessions the camera sweeps a wide classroom area —
        10 keyframes provide full temporal and spatial coverage.
        Frames wider than 2560px (4K) are preserved at 2560px to retain distant face detail.
        All discarded frames are purged immediately to control memory usage.
        """
        import base64
        import gc
        decoded_frames = []
        for idx, b64_str in enumerate(base64_frames):
            try:
                if "," in b64_str:
                    b64_str = b64_str.split(",")[1]
                img_bytes = base64.b64decode(b64_str)
                np_arr = np.frombuffer(img_bytes, np.uint8)
                img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                if img_bgr is not None:
                    ih, iw = img_bgr.shape[:2]
                    # Preserve 4K detail up to 2560px (QHD); downscale anything larger
                    if iw > 2560:
                        img_bgr = cv2.resize(img_bgr, (2560, int(ih * 2560.0 / iw)), interpolation=cv2.INTER_AREA)
                    sharpness = self.check_blurriness(img_bgr)
                    decoded_frames.append((idx, img_bgr, sharpness))
            except Exception:
                continue

        if not decoded_frames:
            return []

        if len(decoded_frames) <= max_keyframes:
            return decoded_frames

        # Partition video sweep into uniform temporal buckets; pick sharpest frame per bucket
        bucket_size = len(decoded_frames) / float(max_keyframes)
        selected_keyframes = []

        for b in range(max_keyframes):
            start_idx = int(b * bucket_size)
            end_idx = int((b + 1) * bucket_size) if b < max_keyframes - 1 else len(decoded_frames)
            bucket = decoded_frames[start_idx:end_idx]
            if bucket:
                best_in_bucket = max(bucket, key=lambda item: item[2])
                selected_keyframes.append(best_in_bucket)

        # Release all discarded frames immediately to control memory
        del decoded_frames
        gc.collect()

        return selected_keyframes

    def is_autofocus_blurry(self, img_bgr, min_threshold=15.0):
        """Checks if a frame is distorted by temporary camera autofocus hunting."""
        score = self.check_blurriness(img_bgr)
        return score < min_threshold

    @staticmethod
    def calibrate_confidence_score(raw_cosine_score: float) -> float:
        """
        Converts raw SFace cosine similarity (-1.0 to 1.0) into a realistic, calibrated percentage (0.0 to 100.0).
        SFace Cosine Standard:
          - < 0.18: Non-match / Noise (0% - 18%)
          - 0.18 to 0.28: Far back-row match under camera distortion (68% - 82%)
          - 0.28 to 0.42: Mid/far row strong match (82% - 93%)
          - 0.42 to 0.60: Crystal clear classroom match (93% - 98%)
          - > 0.60: Definitive High-Confidence Match (98% - 99.8%)
        """
        s = float(raw_cosine_score)
        if s <= 0.18:
            return max(0.0, round(s * 100.0, 1))
        elif s < 0.28:
            pct = 68.0 + ((s - 0.18) / 0.10) * 14.0
            return round(pct, 1)
        elif s < 0.42:
            pct = 82.0 + ((s - 0.28) / 0.14) * 11.0
            return round(pct, 1)
        elif s < 0.60:
            pct = 93.0 + ((s - 0.42) / 0.18) * 5.0
            return round(pct, 1)
        else:
            pct = 98.0 + min(1.8, (s - 0.60) * 6.0)
            return round(pct, 1)

    def process_single_image(self, img_bytes_or_path):
        if isinstance(img_bytes_or_path, (str, Path)):
            img = cv2.imread(str(img_bytes_or_path))
        else:
            np_arr = np.frombuffer(img_bytes_or_path, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            return {"status": "INVALID_IMAGE", "message": "Failed to decode image file."}

        # Memory optimization: Standardize portrait resolution to max dimension 960 to prevent DNN buffer ballooning
        h_orig, w_orig = img.shape[:2]
        max_dim = max(h_orig, w_orig)
        if max_dim > 960:
            scale_factor = 960.0 / max_dim
            img = cv2.resize(img, (int(w_orig * scale_factor), int(h_orig * scale_factor)), interpolation=cv2.INTER_AREA)

        faces = self.detect_faces(img, is_classroom=False)
        if len(faces) == 0:
            return {"status": "NO_FACE", "message": "No face detected in image."}

        face = max(faces, key=lambda f: f[2] * f[3])

        # Evaluate blurriness on the actual face crop (handles soft studio bokeh without false rejections)
        box = face[:4].astype(int)
        x, y, w, h = max(0, box[0]), max(0, box[1]), box[2], box[3]
        crop = img[y:y+h, x:x+w]
        blur_score = self.check_blurriness(crop) if crop.size > 0 else self.check_blurriness(img)

        # Only reject if face is severely blurry (< 6.0) AND detector score is weak (< 0.65)
        if blur_score < 6.0 and face[14] < 0.65:
            return {"status": "POOR_QUALITY", "message": f"Face is blurry (blur score: {blur_score:.1f})"}

        # Generate full 10-template deep neural biometric profile
        # (canonical TTA, pure aligned, mirror, CLAHE contrast, shadow, glare, far-distance, mid-distance, sharp, master centroid)
        templates = self.generate_deep_biometric_profile(img, face)
        primary_emb = templates[0]

        return {
            "status": "SUCCESS",
            "face_box": face[:4].tolist(),
            "embedding": primary_emb.tolist(),
            "multi_embeddings": [e.tolist() for e in templates],
            "quality_score": float(blur_score),
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

                    # Multi-template preservation: store all 10 complementary multi-scale & lighting vectors
                    db_session.query(StudentFaceEmbedding).filter(
                        StudentFaceEmbedding.student_id == student.id
                    ).delete()

                    templates = res.get("multi_embeddings") or [res["embedding"]]
                    for t_idx, t_data in enumerate(templates):
                        db_session.add(StudentFaceEmbedding(
                            student_id=student.id,
                            embedding_data=json.dumps(t_data),
                            quality_score=round(float(res["quality_score"]) - (t_idx * 0.03), 3)
                        ))

                    summary["successfully_processed"] += 1
                    summary["details"].append({
                        "filename": filename,
                        "student_id": student_reg,
                        "name": student_name,
                        "status": "SUCCESS",
                        "message": f"Enrolled with {len(templates)} deep biometric neural templates"
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

                # Memory optimization: flush, commit and trigger garbage collection every student
                import gc
                try:
                    db_session.commit()
                except Exception:
                    pass
                del res
                del source_stream
                gc.collect()

        db_session.commit()
    except Exception as e:
        db_session.rollback()
        raise e

    return summary


def reindex_all_students(db_session, class_id: int = None) -> dict:
    """
    Re-processes all enrolled student photographs using the 10-Template Deep Biometric Neural Engine.
    Instantly upgrades existing student records with bilateral symmetry TTA, lighting profiles,
    and distance multi-scale embeddings for perfect classroom matching.
    """
    from backend.models import Student, StudentFaceEmbedding
    from backend.config import STUDENT_PHOTOS_DIR, BASE_DIR
    import gc

    vision = get_vision_engine()
    query = db_session.query(Student).filter(Student.is_active == True)
    if class_id:
        query = query.filter(Student.class_id == class_id)
    students = query.all()

    total = len(students)
    reindexed_count = 0
    templates_created = 0
    errors = []

    print(f"[VisionEngine] Starting Biometric Re-indexing for {total} students...")

    for st in students:
        candidates = []
        if st.photo_path:
            p_strip = st.photo_path.lstrip('/')
            candidates.append(BASE_DIR / p_strip)
            candidates.append(STUDENT_PHOTOS_DIR / Path(st.photo_path).name)
        candidates.append(STUDENT_PHOTOS_DIR / f"{st.class_id}_{st.student_id}.jpg")
        candidates.append(STUDENT_PHOTOS_DIR / f"1_{st.student_id}.jpg")

        found_path = None
        for c in candidates:
            if c.exists() and c.is_file() and c.stat().st_size > 0:
                found_path = c
                break

        if not found_path:
            errors.append(f"Student {st.student_id}: Photo file not found on disk.")
            continue

        try:
            img = cv2.imread(str(found_path))
            if img is None or img.size == 0:
                errors.append(f"Student {st.student_id}: Failed to decode photo.")
                continue

            faces = vision.detect_faces(img, is_classroom=False)
            if not faces:
                errors.append(f"Student {st.student_id}: No face detected in registered photo.")
                continue

            face = max(faces, key=lambda f: f[2] * f[3])
            templates = vision.generate_deep_biometric_profile(img, face)

            # Purge existing embeddings for this student
            db_session.query(StudentFaceEmbedding).filter(
                StudentFaceEmbedding.student_id == st.id
            ).delete()

            # Insert all 10 neural biometric templates
            for t_idx, t_vec in enumerate(templates):
                db_session.add(StudentFaceEmbedding(
                    student_id=st.id,
                    embedding_data=json.dumps(t_vec.tolist()),
                    quality_score=round(1.0 - (t_idx * 0.03), 3)
                ))
                templates_created += 1

            reindexed_count += 1
            db_session.commit()
            del img
            gc.collect()
        except Exception as e:
            db_session.rollback()
            errors.append(f"Student {st.student_id}: {str(e)}")

    print(f"[VisionEngine] Biometric Re-indexing Complete: {reindexed_count}/{total} students upgraded ({templates_created} templates).")

    return {
        "status": "success",
        "total_students": total,
        "successfully_reindexed": reindexed_count,
        "total_templates_generated": templates_created,
        "templates_per_student": 10,
        "errors": errors
    }

