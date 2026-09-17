import base64
import json
import uuid
import cv2
import numpy as np
from datetime import datetime
from sqlalchemy.orm import Session
from backend.config import (
    THRESHOLD_HIGH_CONFIDENCE, THRESHOLD_MEDIUM_CONFIDENCE,
    REVIEW_CROPS_DIR
)
from backend.models import (
    Student, StudentFaceEmbedding, AttendanceSession,
    AttendanceRecord, AttendanceReview, SessionStatus, AttendanceStatus, Subject
)
from backend.services.vision_service import get_vision_engine


def process_smartboard_session(
    class_id: int,
    subject_id: int = None,
    taken_by_user_id: int = 1,
    base64_frames: list[str] = [],
    db_session: Session = None
) -> dict:
    vision = get_vision_engine()

    if not subject_id and db_session:
        subj = db_session.query(Subject).filter(Subject.class_id == class_id).first()
        if not subj:
            subj = Subject(name="General AI Lecture", code="AI301", class_id=class_id)
            db_session.add(subj)
            db_session.flush()
        subject_id = subj.id

    students = db_session.query(Student).filter(
        Student.class_id == class_id,
        Student.is_active == True
    ).all()

    if not students:
        raise ValueError(f"No active students found in Class ID {class_id}.")

    student_map = {s.id: s for s in students}
    embedding_records = db_session.query(StudentFaceEmbedding).join(Student).filter(
        Student.class_id == class_id
    ).all()

    # Build student ID list and 2D NumPy embedding matrix across multi-templates
    student_ids = []
    ref_vectors = []
    for record in embedding_records:
        try:
            vec = json.loads(record.embedding_data)
            student_ids.append(record.student_id)
            ref_vectors.append(vec)
        except Exception:
            continue

    if ref_vectors:
        ref_matrix = np.array(ref_vectors, dtype=np.float32)
    else:
        ref_matrix = np.empty((0, 128), dtype=np.float32)

    total_frames = len(base64_frames)
    min_required_hits = max(1, int(total_frames * 0.12))  # Optimized hit threshold for classroom detection

    student_evidence = {s.id: {"scores": [], "crops": [], "best_emb": None} for s in students}
    unrecognized_crops = []
    frame_overlay_boxes = []

    for frame_idx, b64_str in enumerate(base64_frames):
        try:
            if "," in b64_str:
                b64_str = b64_str.split(",")[1]
            img_bytes = base64.b64decode(b64_str)
            np_arr = np.frombuffer(img_bytes, np.uint8)
            img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if img_bgr is None:
                continue

            # Autofocus Hunting & Motion Blur Filtering
            if vision.is_autofocus_blurry(img_bgr, min_threshold=12.0) and total_frames > 3:
                # Skip frames severely distorted by lens focus hunting
                print(f"[AttendanceService] Skipping frame {frame_idx} due to camera focus blur.")
                continue

            faces = vision.detect_faces(img_bgr)
            if len(faces) == 0:
                continue

            frame_candidates = []

            for face in faces:
                emb = vision.extract_embedding(img_bgr, face)
                box = face[:4].astype(int)
                x, y, w, h = box
                x, y = max(0, x), max(0, y)

                h_img, w_img, _ = img_bgr.shape
                crop = img_bgr[y:min(y+h, h_img), x:min(x+w, w_img)]
                local_sharpness = vision.calculate_crop_sharpness(img_bgr, [x, y, w, h])

                best_student_id = None
                best_score = -1.0

                # Multi-Template Vectorized Cosine Similarity
                if ref_matrix.size > 0:
                    scores_array = vision.compare_embeddings_batch(emb, ref_matrix)
                    student_max_scores = {}
                    for idx, score_val in enumerate(scores_array):
                        st_id = student_ids[idx]
                        if st_id not in student_max_scores or score_val > student_max_scores[st_id]:
                            student_max_scores[st_id] = float(score_val)

                    if student_max_scores:
                        top_st_id, top_score = max(student_max_scores.items(), key=lambda item: item[1])
                        if top_score >= THRESHOLD_MEDIUM_CONFIDENCE:
                            best_student_id = top_st_id
                            best_score = top_score

                crop_filename = f"crop_{uuid.uuid4().hex[:10]}.jpg"
                crop_path = REVIEW_CROPS_DIR / crop_filename
                crop_url = f"/static/uploads/review_crops/{crop_filename}"
                
                if crop.size > 0:
                    cv2.imwrite(str(crop_path), crop)

                frame_candidates.append({
                    "face_box": [int(x), int(y), int(w), int(h)],
                    "best_student_id": best_student_id,
                    "score": best_score,
                    "sharpness": local_sharpness,
                    "crop_url": crop_url,
                    "emb": emb
                })

            # Non-Maximum Identity Suppression (1 Student ID per frame max)
            assigned_students_this_frame = set()
            frame_boxes = []

            # Sort candidate face recognitions by score descending
            frame_candidates.sort(key=lambda c: c["score"], reverse=True)

            for cand in frame_candidates:
                st_id = cand["best_student_id"]
                score = cand["score"]
                crop_url = cand["crop_url"]
                box = cand["face_box"]
                sharpness = cand.get("sharpness", 0.0)

                calibrated_pct = vision.calibrate_confidence_score(score)

                if st_id and st_id not in assigned_students_this_frame and score >= THRESHOLD_MEDIUM_CONFIDENCE:
                    assigned_students_this_frame.add(st_id)
                    st_obj = student_map[st_id]
                    st_label = f"{st_obj.student_id} ({calibrated_pct:.0f}%)"

                    if "sharpnesses" not in student_evidence[st_id]:
                        student_evidence[st_id]["sharpnesses"] = []

                    student_evidence[st_id]["scores"].append(score)
                    student_evidence[st_id]["crops"].append(crop_url)
                    student_evidence[st_id]["sharpnesses"].append(sharpness)

                    # Multi-Focal Depth Peak Selection: Lock crop & embedding from peak-sharpness focal frame
                    best_sharp = max(student_evidence[st_id]["sharpnesses"])
                    if sharpness >= best_sharp or student_evidence[st_id]["best_emb"] is None:
                        student_evidence[st_id]["best_emb"] = cand["emb"]
                        student_evidence[st_id]["peak_crop"] = crop_url
                else:
                    st_label = "Unknown"
                    unrecognized_crops.append({"crop_url": crop_url, "score": score})

                frame_boxes.append({
                    "box": box,
                    "label": st_label,
                    "score": calibrated_pct
                })

            frame_overlay_boxes.append({"frame_index": frame_idx, "boxes": frame_boxes})

        except Exception as e:
            print(f"[AttendanceService] Error processing frame {frame_idx}: {e}")
            continue

    now = datetime.utcnow()
    session_date = now.strftime("%Y-%m-%d")
    session_time = now.strftime("%H:%M:%S")

    att_session = AttendanceSession(
        class_id=class_id,
        subject_id=subject_id,
        taken_by_user_id=taken_by_user_id,
        session_date=session_date,
        session_time=session_time,
        status=SessionStatus.PENDING_REVIEW.value,
        total_students=len(students)
    )
    db_session.add(att_session)
    db_session.flush()

    present_cnt = 0
    absent_cnt = 0
    pending_cnt = 0
    record_items = []

    for student in students:
        ev = student_evidence[student.id]
        scores = ev["scores"]
        crops = ev["crops"]

        frame_hits = len(scores)
        max_score = max(scores) if scores else 0.0
        calibrated_pct = vision.calibrate_confidence_score(max_score) if scores else 0.0

        status = AttendanceStatus.ABSENT.value

        if max_score >= THRESHOLD_HIGH_CONFIDENCE and frame_hits >= min_required_hits:
            status = AttendanceStatus.PRESENT.value
            present_cnt += 1
            # Online Embedding Auto-Enrichment (Continuous Model Enhancement)
            if ev.get("best_emb") is not None and max_score >= 0.50:
                try:
                    curr_emb_count = db_session.query(StudentFaceEmbedding).filter(
                        StudentFaceEmbedding.student_id == student.id
                    ).count()
                    if curr_emb_count < 10:
                        enrich_rec = StudentFaceEmbedding(
                            student_id=student.id,
                            embedding_data=json.dumps(ev["best_emb"].tolist()),
                            quality_score=float(max_score)
                        )
                        db_session.add(enrich_rec)
                except Exception as ex:
                    print(f"[Auto-Enrich] Model enrichment skipped: {ex}")
        elif max_score >= THRESHOLD_MEDIUM_CONFIDENCE:
            status = AttendanceStatus.REVIEW.value
            pending_cnt += 1
            review_entry = AttendanceReview(
                session_id=att_session.id,
                student_id=student.id,
                captured_face_crop=ev.get("peak_crop") or (crops[0] if crops else student.photo_path),
                match_score=calibrated_pct,
                review_status="PENDING"
            )
            db_session.add(review_entry)
        else:
            status = AttendanceStatus.ABSENT.value
            absent_cnt += 1

        record = AttendanceRecord(
            session_id=att_session.id,
            student_id=student.id,
            status=status,
            confidence=round(calibrated_pct, 2),
            notes=f"Auto-recognized ({frame_hits} frames)" if status == AttendanceStatus.PRESENT.value else None
        )
        db_session.add(record)
        record_items.append({
            "student_id": student.student_id,
            "student_name": student.name,
            "status": status,
            "confidence": f"{round(calibrated_pct, 1)}%",
            "registered_photo": student.photo_path,
            "captured_crop": ev.get("peak_crop") or (crops[0] if crops else None)
        })

    att_session.present_count = present_cnt
    att_session.absent_count = absent_cnt
    att_session.pending_count = pending_cnt

    db_session.commit()

    return {
        "session_id": att_session.id,
        "date": session_date,
        "time": session_time,
        "total_students": len(students),
        "present_count": present_cnt,
        "pending_count": pending_cnt,
        "absent_count": absent_cnt,
        "status": att_session.status,
        "records": record_items,
        "frame_overlays": frame_overlay_boxes,
        "unrecognized_faces_count": len(unrecognized_crops)
    }


def confirm_staff_reviews(
    session_id: int,
    confirmations: list[dict],
    confirmed_by_user_id: int,
    db_session: Session
):
    session = db_session.query(AttendanceSession).filter(
        AttendanceSession.id == session_id
    ).first()

    if not session:
        raise ValueError(f"Attendance session {session_id} not found.")

    now = datetime.utcnow()

    for item in confirmations:
        st_id = item["student_id"]
        new_status = item["status"]

        record = db_session.query(AttendanceRecord).filter(
            AttendanceRecord.session_id == session_id,
            AttendanceRecord.student_id == st_id
        ).first()

        if record:
            record.status = new_status
            record.confirmed_by_user_id = confirmed_by_user_id
            record.confirmation_time = now
            record.notes = f"Staff confirmed as {new_status}"

        review = db_session.query(AttendanceReview).filter(
            AttendanceReview.session_id == session_id,
            AttendanceReview.student_id == st_id
        ).first()

        if review:
            review.review_status = "APPROVED_PRESENT" if new_status == AttendanceStatus.PRESENT.value else "MARKED_ABSENT"

    all_records = db_session.query(AttendanceRecord).filter(
        AttendanceRecord.session_id == session_id
    ).all()

    present_cnt = sum(1 for r in all_records if r.status == AttendanceStatus.PRESENT.value)
    absent_cnt = sum(1 for r in all_records if r.status == AttendanceStatus.ABSENT.value)
    pending_cnt = sum(1 for r in all_records if r.status == AttendanceStatus.REVIEW.value)

    session.present_count = present_cnt
    session.absent_count = absent_cnt
    session.pending_count = pending_cnt
    session.status = SessionStatus.CONFIRMED.value if pending_cnt == 0 else SessionStatus.PENDING_REVIEW.value
    session.confirmed_by_user_id = confirmed_by_user_id
    session.confirmed_at = now

    db_session.commit()
    return session
