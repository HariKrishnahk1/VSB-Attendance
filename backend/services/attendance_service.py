import base64
import gc
import json
import uuid
import cv2
import numpy as np
from datetime import datetime
from sqlalchemy.orm import Session
from backend.config import (
    THRESHOLD_HIGH_CONFIDENCE, THRESHOLD_MEDIUM_CONFIDENCE,
    THRESHOLD_AMBIGUITY_MARGIN, MAX_STUDENT_TEMPLATES,
    REVIEW_CROPS_DIR
)
from backend.models import (
    Student, StudentFaceEmbedding, AttendanceSession,
    AttendanceRecord, AttendanceReview, SessionStatus, AttendanceStatus, Subject
)
from backend.services.vision_service import get_vision_engine


def _solve_bipartite_matching(cost_matrix):
    """
    Solves linear sum assignment (Hungarian matching) using pure NumPy Jonker-Volgenant algorithm.
    Zero external dependencies, zero memory overhead, 100% mathematically optimal.
    """
    C = np.array(cost_matrix, dtype=float)
    n_rows, n_cols = C.shape
    transpose = n_rows > n_cols
    if transpose:
        C = C.T
        n_rows, n_cols = C.shape

    u = np.zeros(n_rows + 1)
    v = np.zeros(n_cols + 1)
    p = np.zeros(n_cols + 1, dtype=int)
    way = np.zeros(n_cols + 1, dtype=int)

    for i in range(1, n_rows + 1):
        p[0] = i
        j0 = 0
        minv = np.full(n_cols + 1, np.inf)
        used = np.zeros(n_cols + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = np.inf
            j1 = 0
            for j in range(1, n_cols + 1):
                if not used[j]:
                    cur = C[i0 - 1, j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(n_cols + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    row_ind = np.zeros(n_rows, dtype=int)
    col_ind = np.zeros(n_rows, dtype=int)
    for j in range(1, n_cols + 1):
        if p[j] != 0 and p[j] <= n_rows:
            row_ind[p[j] - 1] = p[j] - 1
            col_ind[p[j] - 1] = j - 1

    if transpose:
        order = np.argsort(col_ind)
        return col_ind[order], row_ind[order]
    return row_ind, col_ind


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
        # Smart Fallback: Auto-resolve to the class that contains registered students
        alt_student = db_session.query(Student).filter(Student.is_active == True).first()
        if alt_student:
            class_id = alt_student.class_id
            students = db_session.query(Student).filter(
                Student.class_id == class_id,
                Student.is_active == True
            ).all()

    if not students:
        raise ValueError("No registered students found in database. Please upload a student dataset .ZIP file in the Admin Portal.")

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

    # Build pre-indexed student template map for instant vectorized scoring
    student_template_indices = {}
    for idx, sid in enumerate(student_ids):
        if sid not in student_template_indices:
            student_template_indices[sid] = []
        student_template_indices[sid].append(idx)
    student_template_indices = {sid: np.array(idxs, dtype=int) for sid, idxs in student_template_indices.items()}

    # Keyframe count is env-aware: 2 on Render (512MB), 8 on local/smartboard (8GB)
    selected_keyframes = vision.select_focal_keyframes(base64_frames)
    if not selected_keyframes:
        selected_keyframes = []

    total_frames = len(selected_keyframes)

    student_evidence = {s.id: {"scores": [], "crops": [], "best_emb": None, "sharpnesses": [], "margins": [], "is_distant": False} for s in students}
    unrecognized_crops = []
    frame_overlay_boxes = []

    def _process_frame_worker(kf_item):
        frame_idx, img_bgr, frame_sharpness = kf_item
        if img_bgr is None or img_bgr.size == 0:
            return frame_idx, []

        faces = vision.detect_faces(img_bgr, is_classroom=True)
        if not faces:
            return frame_idx, []

        face_meta = []
        face_embs = []
        h_img, w_img, _ = img_bgr.shape

        for face in faces:
            emb = vision.extract_embedding(img_bgr, face, use_tta=True)
            box = face[:4].astype(int)
            x, y, w, h = box
            x, y = max(0, x), max(0, y)
            crop = img_bgr[y:min(y+h, h_img), x:min(x+w, w_img)]
            local_sharpness = vision.calculate_crop_sharpness(img_bgr, [x, y, w, h])

            face_meta.append({
                "face_box": [int(x), int(y), int(w), int(h)],
                "sharpness": local_sharpness,
                "is_distant": (w < 90 or h < 90),  # 4K: mid-row faces are ~80-120px
                "crop": crop,
                "crop_url": None,
                "emb": emb
            })
            face_embs.append(emb)

        if not face_embs or ref_matrix.size == 0:
            return frame_idx, []

        def _save_crop_if_needed(meta_item):
            if meta_item["crop_url"]:
                return meta_item["crop_url"]
            c = meta_item.get("crop")
            if c is not None and c.size > 0:
                crop_filename = f"crop_{uuid.uuid4().hex[:10]}.jpg"
                crop_path = REVIEW_CROPS_DIR / crop_filename
                cv2.imwrite(str(crop_path), c)
                meta_item["crop_url"] = f"/static/uploads/review_crops/{crop_filename}"
            return meta_item["crop_url"]

        # Construct Similarity Matrix between Detected Faces and Registered Students
        # Shape: (num_faces, num_students)
        num_faces = len(face_embs)
        num_students = len(students)
        student_obj_ids = [s.id for s in students]
        sim_matrix = np.zeros((num_faces, num_students), dtype=np.float32)

        for f_idx, emb in enumerate(face_embs):
            scores_array = vision.compare_embeddings_batch(emb, ref_matrix)
            for s_col, s_id in enumerate(student_obj_ids):
                idxs = student_template_indices.get(s_id)
                if idxs is not None and len(idxs) > 0:
                    s_scores = scores_array[idxs]
                    s_sorted = np.sort(s_scores)[::-1]
                    # Weighted combination: 70% top-1, 20% top-2, 10% top-3
                    t1 = float(s_sorted[0])
                    t2 = float(s_sorted[1]) if len(s_sorted) > 1 else t1
                    t3 = float(s_sorted[2]) if len(s_sorted) > 2 else t2
                    comp_score = 0.70 * t1 + 0.20 * t2 + 0.10 * t3
                    sim_matrix[f_idx, s_col] = comp_score

        # Solve Global Optimal Bipartite Matching (Maximum Weight Assignment)
        row_ind, col_ind = _solve_bipartite_matching(-sim_matrix)

        candidates = []
        assigned_faces = set()

        for r, c in zip(row_ind, col_ind):
            assigned_faces.add(r)
            st_id = student_obj_ids[c]
            match_score = float(sim_matrix[r, c])
            meta = face_meta[r]
            is_dist = meta["is_distant"]

            # Compute margin against runner-up student for this face
            row_scores = sorted(sim_matrix[r, :], reverse=True)
            margin = (row_scores[0] - row_scores[1]) if len(row_scores) > 1 else match_score

            # Adaptive thresholds tuned for 4K smartboard face sizes
            face_w, face_h = meta["face_box"][2], meta["face_box"][3]
            if face_w < 20 or face_h < 20:      # extreme back-row (ultra-small, 4K specific)
                min_match_thresh = 0.33
                min_margin_thresh = 0.015
            elif face_w < 50 or face_h < 50:    # back-row small face
                min_match_thresh = 0.36
                min_margin_thresh = 0.018
            elif face_w < 90 or face_h < 90:    # mid-row
                min_match_thresh = 0.40
                min_margin_thresh = 0.022
            else:                                # front/mid-row (clear face)
                min_match_thresh = 0.46
                min_margin_thresh = 0.028

            is_valid_student = (match_score >= min_match_thresh and margin >= min_margin_thresh)

            # Lazy write crop only if student was matched or close candidate
            c_url = _save_crop_if_needed(meta) if (is_valid_student or match_score >= 0.30) else None

            candidates.append({
                "face_box": meta["face_box"],
                "best_student_id": st_id if is_valid_student else None,
                "score": match_score if is_valid_student else -1.0,
                "margin": margin,
                "sharpness": meta["sharpness"],
                "is_distant": is_dist,
                "crop_url": c_url,
                "emb": meta["emb"]
            })

        # Include any unassigned faces (if num_faces > num_students)
        for r in range(num_faces):
            if r not in assigned_faces:
                meta = face_meta[r]
                candidates.append({
                    "face_box": meta["face_box"],
                    "best_student_id": None,
                    "score": -1.0,
                    "margin": 0.0,
                    "sharpness": meta["sharpness"],
                    "is_distant": meta["is_distant"],
                    "crop_url": None,
                    "emb": meta["emb"]
                })

        # Explicit cleanup of per-frame arrays
        for m in face_meta:
            m["crop"] = None
        del face_embs, sim_matrix, face_meta
        gc.collect()
        return frame_idx, candidates

    # High-performance sequential execution utilizing OpenCV's native thread pool
    frame_results = []
    for kf_item in selected_keyframes:
        frame_results.append(_process_frame_worker(kf_item))
    del selected_keyframes
    gc.collect()

    # Sort results by frame index order
    frame_results.sort(key=lambda r: r[0])

    for frame_idx, frame_candidates in frame_results:
        assigned_students_this_frame = set()
        frame_boxes = []

        # Sort candidate face recognitions by score descending
        frame_candidates.sort(key=lambda c: c["score"], reverse=True)

        for cand in frame_candidates:
            st_id = cand["best_student_id"]
            score = cand["score"]
            margin = cand.get("margin", 0.0)
            crop_url = cand["crop_url"]
            box = cand["face_box"]
            sharpness = cand.get("sharpness", 0.0)
            is_dist = cand.get("is_distant", False)

            calibrated_pct = vision.calibrate_confidence_score(score) if score > 0 else 0.0

            # Threshold check matching per-size values set in the worker
            face_w_ev = box[2] if box else 100
            if face_w_ev < 20:
                ev_thresh = 0.33
            elif face_w_ev < 50:
                ev_thresh = 0.36
            elif face_w_ev < 90:
                ev_thresh = 0.40
            else:
                ev_thresh = 0.46

            if st_id and st_id not in assigned_students_this_frame and score >= ev_thresh:
                assigned_students_this_frame.add(st_id)
                st_obj = student_map[st_id]
                st_label = f"{st_obj.student_id} ({calibrated_pct:.0f}%)"

                student_evidence[st_id]["scores"].append(score)
                student_evidence[st_id]["crops"].append(crop_url)
                student_evidence[st_id]["sharpnesses"].append(sharpness)
                student_evidence[st_id]["margins"].append(margin)
                if is_dist:
                    student_evidence[st_id]["is_distant"] = True

                # Peak Sharpness Focal Selection
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
        max_margin = max(ev.get("margins", [0.0])) if scores else 0.0
        calibrated_pct = vision.calibrate_confidence_score(max_score) if scores else 0.0

        is_distant = ev.get("is_distant", False)

        # -----------------------------------------------------------------------
        # ATTENDANCE DECISION LADDER (SFace cosine, calibrated thresholds)
        # -----------------------------------------------------------------------
        # Tier 1 – HIGH CONFIDENCE (front/mid row, score >= 0.50, any frame count)
        #          -> PRESENT immediately
        # Tier 2 – MEDIUM CONFIDENCE (score >= 0.42, 2+ frame hits)
        #          -> PRESENT (multi-frame consensus confirms identity)
        # Tier 3 – DISTANT / BACK-ROW (face < 70px, score >= 0.38, 2+ hits)
        #          -> PRESENT (degraded image, consensus required)
        # Tier 4 – BORDERLINE (score >= 0.34, 3+ hits)
        #          -> REVIEW (flag for staff confirmation)
        # Tier 5 – LOW / NO MATCH (score < 0.34 or only 1 weak hit)
        #          -> ABSENT
        # -----------------------------------------------------------------------
        is_present = (
            (max_score >= 0.50 and frame_hits >= 1) or                          # Tier 1
            (max_score >= 0.42 and frame_hits >= 2) or                          # Tier 2
            (is_distant and max_score >= 0.38 and frame_hits >= 2) or           # Tier 3
            (max_score >= 0.40 and frame_hits >= 3)                             # Tier 2b (3-frame lower score)
        )
        is_review = (
            not is_present and
            max_score >= 0.34 and
            frame_hits >= 2
        )

        if is_present:
            status = AttendanceStatus.PRESENT.value
            present_cnt += 1
            # Online Embedding Auto-Enrichment only when score is very high (genuine match only)
            if ev.get("best_emb") is not None and max_score >= 0.52:
                try:
                    curr_emb_count = db_session.query(StudentFaceEmbedding).filter(
                        StudentFaceEmbedding.student_id == student.id
                    ).count()
                    if curr_emb_count < MAX_STUDENT_TEMPLATES:
                        enrich_rec = StudentFaceEmbedding(
                            student_id=student.id,
                            embedding_data=json.dumps(ev["best_emb"].tolist()),
                            quality_score=float(max_score)
                        )
                        db_session.add(enrich_rec)
                except Exception as ex:
                    print(f"[Auto-Enrich] Model enrichment skipped: {ex}")
        elif is_review:
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

        note = None
        if status == AttendanceStatus.PRESENT.value:
            if is_distant:
                note = f"Auto-recognized ({frame_hits} frames, Back-Row Biometric Match)"
            else:
                note = f"Auto-recognized ({frame_hits} frames)"

        record = AttendanceRecord(
            session_id=att_session.id,
            student_id=student.id,
            status=status,
            confidence=round(calibrated_pct, 2),
            notes=note
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
    att_session.status = SessionStatus.CONFIRMED.value if pending_cnt == 0 else SessionStatus.PENDING_REVIEW.value

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
