"""
VSB Attendance System - Comprehensive Face Recognition & Attendance Accuracy Tests
==================================================================================
Tests cover:
  1. Authentication (all 4 roles)
  2. DB enrollment health (students + embeddings exist)
  3. Self-match probe accuracy (registered photo must match itself at >= 0.75 cosine)
  4. Cross-student separation (different students must score < 0.42 against each other)
  5. End-to-end smartboard session with real student photo frames
  6. Attendance count accuracy (present/absent must add up to total_students)
  7. Staff review + Excel generation
  8. HOD dashboard
"""

import sys
import os
import json
import base64
import cv2
import numpy as np
from pathlib import Path
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from main import app
from backend.database import SessionLocal
from backend.models import Student, AttendanceSession, AttendanceRecord, User, StudentFaceEmbedding
from backend.services.vision_service import get_vision_engine
from backend.config import STUDENT_PHOTOS_DIR

client = TestClient(app)

import pytest


# ===========================================================================
# Helper utilities
# ===========================================================================

def get_tokens():
    roles = [
        ("admin",         "admin123",  "ADMIN"),
        ("hod_aids",      "hod123",    "HOD"),
        ("staff_hari",    "staff123",  "STAFF"),
        ("class_3aids_a", "class123",  "CLASS"),
    ]
    tokens = {}
    for username, password, expected_role in roles:
        res = client.post("/api/auth/login", json={"username": username, "password": password})
        assert res.status_code == 200, f"Login failed for {username}: {res.text}"
        data = res.json()
        assert data["role"] == expected_role
        tokens[expected_role] = data["access_token"]
        print(f"  [OK] Login: {username} ({expected_role})")
    return tokens


def photo_to_b64(photo_path):
    img = cv2.imread(str(photo_path))
    if img is None:
        raise FileNotFoundError(f"Could not read: {photo_path}")
    img = cv2.resize(img, (640, 480), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        raise RuntimeError(f"Failed to encode photo: {photo_path}")
    b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def get_registered_student_photos(limit=6):
    db = SessionLocal()
    try:
        students = db.query(Student).filter(Student.is_active == True).limit(limit).all()
        results = []
        for st in students:
            photo_path = STUDENT_PHOTOS_DIR / f"{st.class_id}_{st.student_id}.jpg"
            if not photo_path.exists() and st.photo_path:
                alt = Path(st.photo_path.lstrip("/"))
                if alt.exists():
                    photo_path = alt
            if photo_path.exists():
                results.append((st, photo_path))
        return results
    finally:
        db.close()


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture(scope="module")
def tokens():
    return get_tokens()


def create_realistic_session(tokens):
    print("\n--- Creating Realistic Attendance Session with Real Photos ---")
    class_headers = {"Authorization": f"Bearer {tokens['CLASS']}"}
    classes = client.get("/api/admin/classes").json()
    assert classes, "No classes in DB"
    class_id = classes[0]["id"]

    student_photos = get_registered_student_photos(limit=10)
    if student_photos:
        frames = []
        for st, photo_path in student_photos:
            try:
                b64_frame = photo_to_b64(photo_path)
                frames.append(b64_frame)
                frames.append(b64_frame)
            except Exception as e:
                print(f"  [WARN] Could not encode {st.student_id}: {e}")
        print(f"  Using {len(frames)} real photo frames ({len(student_photos)} unique students)")
    else:
        print("  [WARN] No student photos on disk - using gray fallback frames")
        blank = np.full((480, 640, 3), 128, dtype=np.uint8)
        ok, buf = cv2.imencode(".jpg", blank)
        b64_blank = f"data:image/jpeg;base64,{base64.b64encode(buf.tobytes()).decode()}"
        frames = [b64_blank] * 6

    res = client.post(
        "/api/attendance/process-frames",
        json={"class_id": class_id, "frames": frames},
        headers=class_headers
    )
    assert res.status_code == 200, f"Process frames failed: {res.text}"
    session_data = res.json()
    sid = session_data["session_id"]
    print(f"  Session #{sid}: total={session_data.get('total_students')}, present={session_data.get('present_count')}, absent={session_data.get('absent_count')}, pending={session_data.get('pending_count')}")
    return sid


@pytest.fixture(scope="module")
def session_id(tokens):
    return create_realistic_session(tokens)


# ===========================================================================
# Tests
# ===========================================================================

def test_auth():
    print("\n--- Test 1: Authentication ---")
    t = get_tokens()
    assert set(t.keys()) == {"ADMIN", "HOD", "STAFF", "CLASS"}
    print("  [PASS] All roles authenticated")


def test_dataset_enrollment():
    print("\n--- Test 2: Enrollment Health Check ---")
    db = SessionLocal()
    try:
        student_count = db.query(Student).filter(Student.is_active == True).count()
        embedding_count = db.query(StudentFaceEmbedding).count()
        print(f"  Active students  : {student_count}")
        print(f"  Total embeddings : {embedding_count}")
        assert student_count > 0, "No active students in DB! Upload a ZIP first."
        assert embedding_count > 0, "No embeddings in DB! Re-upload or run reindex."
        assert embedding_count >= student_count, f"embedding_count ({embedding_count}) < student_count ({student_count})"
        print(f"  Avg templates/student: {embedding_count/student_count:.1f}")
        print(f"  [PASS] Enrollment healthy")
    finally:
        db.close()


def test_self_match_accuracy():
    """
    Self-match probe: each registered photo must match its own stored embedding
    at cosine >= 0.75. Validates enrollment pipeline correctness.
    """
    print("\n--- Test 3: Self-Match Probe Accuracy ---")
    db = SessionLocal()
    vision = get_vision_engine()
    MIN_SELF_MATCH_SCORE = 0.75
    MIN_PASS_RATE = 0.85
    try:
        students = db.query(Student).filter(Student.is_active == True).limit(20).all()
        if not students:
            pytest.skip("No students enrolled")

        passed = 0
        failed_list = []

        for st in students:
            photo_path = STUDENT_PHOTOS_DIR / f"{st.class_id}_{st.student_id}.jpg"
            if not photo_path.exists() and st.photo_path:
                alt = Path(st.photo_path.lstrip("/"))
                if alt.exists():
                    photo_path = alt
            if not photo_path.exists():
                continue

            img = cv2.imread(str(photo_path))
            if img is None:
                continue

            faces = vision.detect_faces(img, is_classroom=False)
            if not faces:
                print(f"  [WARN] {st.student_id}: no face detected in registered photo")
                continue

            face = max(faces, key=lambda f: f[2] * f[3])
            live_emb = vision.extract_embedding(img, face, use_tta=True)
            live_emb = live_emb / (np.linalg.norm(live_emb) + 1e-9)

            stored_embs = db.query(StudentFaceEmbedding).filter(
                StudentFaceEmbedding.student_id == st.id
            ).all()
            if not stored_embs:
                continue

            best_score = -1.0
            for se in stored_embs:
                try:
                    vec = np.array(json.loads(se.embedding_data), dtype=np.float32)
                    vec = vec / (np.linalg.norm(vec) + 1e-9)
                    score = float(np.dot(live_emb, vec))
                    best_score = max(best_score, score)
                except Exception:
                    continue

            if best_score >= MIN_SELF_MATCH_SCORE:
                passed += 1
                print(f"  [OK]   {st.student_id}: self-match = {best_score:.4f}")
            else:
                failed_list.append((st.student_id, st.name, best_score))
                print(f"  [FAIL] {st.student_id}: self-match = {best_score:.4f} (< {MIN_SELF_MATCH_SCORE})")

        total_tested = passed + len(failed_list)
        if total_tested == 0:
            pytest.skip("No photos found on disk")

        pass_rate = passed / total_tested
        print(f"\n  Self-Match: {passed}/{total_tested} passed ({pass_rate*100:.1f}%) - min required: {MIN_PASS_RATE*100:.0f}%")
        assert pass_rate >= MIN_PASS_RATE, (
            f"Self-match accuracy too low: {pass_rate*100:.1f}% < {MIN_PASS_RATE*100:.0f}%.\n"
            f"Failed students: {[(s, sc) for s, _, sc in failed_list]}\n"
            "Fix: Re-upload student ZIP or run Admin > Reindex Biometrics."
        )
        print(f"  [PASS] Self-match accuracy: {pass_rate*100:.1f}%")
    finally:
        db.close()


def test_cross_student_separation():
    """
    Cross-student separation: different students must NOT score >= 0.42 against each other.
    Prevents false-positive attendance (Student A marked as Student B).
    """
    print("\n--- Test 4: Cross-Student Separation ---")
    db = SessionLocal()
    vision = get_vision_engine()
    MAX_CROSS_MATCH_SCORE = 0.42
    try:
        students = db.query(Student).filter(Student.is_active == True).limit(10).all()
        if len(students) < 2:
            pytest.skip("Need at least 2 students")

        emb_map = {}
        for st in students:
            recs = db.query(StudentFaceEmbedding).filter(
                StudentFaceEmbedding.student_id == st.id
            ).limit(3).all()
            vecs = []
            for r in recs:
                try:
                    v = np.array(json.loads(r.embedding_data), dtype=np.float32)
                    v = v / (np.linalg.norm(v) + 1e-9)
                    vecs.append(v)
                except Exception:
                    continue
            if vecs:
                emb_map[st.id] = (st.student_id, vecs)

        violations = []
        pairs_tested = 0
        id_list = list(emb_map.keys())
        for i in range(len(id_list)):
            for j in range(i + 1, len(id_list)):
                sid_a, vecs_a = emb_map[id_list[i]]
                sid_b, vecs_b = emb_map[id_list[j]]
                cross_score = float(np.dot(vecs_a[0], vecs_b[0]))
                pairs_tested += 1
                if cross_score >= MAX_CROSS_MATCH_SCORE:
                    violations.append((sid_a, sid_b, cross_score))
                    print(f"  [VIOLATION] {sid_a} <-> {sid_b}: {cross_score:.4f}")

        print(f"  Cross-match pairs tested: {pairs_tested}, violations: {len(violations)}")
        assert len(violations) == 0, (
            f"Identity leakage! {len(violations)} student pairs score >= {MAX_CROSS_MATCH_SCORE}.\n"
            f"Violations: {violations}\n"
            "These students may be mis-identified as each other."
        )
        print(f"  [PASS] Cross-student separation OK ({pairs_tested} pairs tested)")
    finally:
        db.close()


def test_attendance_count_integrity(tokens, session_id):
    """present + absent + pending must equal total_students."""
    print("\n--- Test 5: Attendance Count Integrity ---")
    staff_headers = {"Authorization": f"Bearer {tokens['STAFF']}"}
    detail = client.get(f"/api/attendance/session/{session_id}", headers=staff_headers).json()
    total   = detail["total_students"]
    present = detail["present_count"]
    absent  = detail["absent_count"]
    pending = detail["pending_count"]
    records = detail["records"]

    print(f"  total={total}, present={present}, absent={absent}, pending={pending}, records={len(records)}")
    assert present + absent + pending == total, (
        f"Count mismatch: {present}+{absent}+{pending} != {total}"
    )
    assert len(records) == total, f"Record rows ({len(records)}) != total ({total})"
    valid = {"PRESENT", "ABSENT", "REVIEW"}
    for r in records:
        assert r["status"] in valid, f"Invalid status '{r['status']}' for {r['student_id']}"
    print(f"  [PASS] {present}P + {absent}A + {pending}R = {total} total")


def test_recognition_accuracy(tokens, session_id):
    """At least half of submitted student photos must be recognized as PRESENT or REVIEW."""
    print("\n--- Test 6: Face Recognition Accuracy ---")
    staff_headers = {"Authorization": f"Bearer {tokens['STAFF']}"}
    detail = client.get(f"/api/attendance/session/{session_id}", headers=staff_headers).json()
    present = detail["present_count"]
    pending = detail["pending_count"]
    recognized = present + pending

    student_photos = get_registered_student_photos(limit=10)
    photos_used = len(student_photos)
    print(f"  Photos as frames: {photos_used}, recognized: {recognized}")

    if photos_used > 0:
        min_expected = max(1, photos_used // 2)
        assert recognized >= min_expected, (
            f"Recognition too low: {recognized}/{photos_used} (expected >= {min_expected}).\n"
            "Check threshold calibration, photo quality, or run Admin > Reindex Biometrics."
        )
        print(f"  [PASS] Recognized {recognized}/{photos_used} students")
    else:
        print("  [SKIP] No photos on disk - cannot validate recognition accuracy")


def test_staff_review_and_excel(tokens, session_id):
    """Staff confirmation and Excel download."""
    print("\n--- Test 7: Staff Confirmation & Excel ---")
    staff_headers = {"Authorization": f"Bearer {tokens['STAFF']}"}
    detail = client.get(f"/api/attendance/session/{session_id}", headers=staff_headers).json()

    confirmations = [
        {
            "student_id": r["student_db_id"],
            "status": r["status"] if r["status"] in ("PRESENT", "ABSENT") else "ABSENT"
        }
        for r in detail["records"]
    ]
    conf_res = client.post(
        "/api/attendance/confirm-review",
        json={"session_id": session_id, "confirmations": confirmations},
        headers=staff_headers
    )
    assert conf_res.status_code == 200, f"Confirmation failed: {conf_res.text}"
    print("  [OK] Confirmation saved")

    excel_res = client.post(f"/api/attendance/generate-excel/{session_id}", headers=staff_headers)
    assert excel_res.status_code == 200, f"Excel failed: {excel_res.text}"
    excel_info = excel_res.json()

    download_res = client.get(excel_info["download_url"], headers=staff_headers)
    assert download_res.status_code == 200
    assert len(download_res.content) > 1000, "Excel file too small or corrupt"
    print(f"  [PASS] Excel downloaded: {excel_info['file_name']} ({len(download_res.content)} bytes)")


def test_hod_dashboard(tokens):
    """HOD dashboard must return valid department data."""
    print("\n--- Test 8: HOD Dashboard ---")
    hod_headers = {"Authorization": f"Bearer {tokens['HOD']}"}
    res = client.get("/api/hod/dashboard", headers=hod_headers)
    assert res.status_code == 200, f"HOD dashboard failed: {res.text}"
    data = res.json()
    print(f"  Department: {data['department_name']}, Classes: {data['total_classes']}, Attendance: {data['overall_percentage']}")
    print("  [PASS] HOD dashboard OK")


# ===========================================================================
# Standalone runner
# ===========================================================================

if __name__ == "__main__":
    print("=" * 65)
    print("  VSB Attendance - Face Recognition Accuracy Test Suite")
    print("=" * 65)
    t = get_tokens()
    test_dataset_enrollment()
    test_self_match_accuracy()
    test_cross_student_separation()
    sid = create_realistic_session(t)
    test_attendance_count_integrity(t, sid)
    test_recognition_accuracy(t, sid)
    test_staff_review_and_excel(t, sid)
    test_hod_dashboard(t)
    print("\n" + "=" * 65)
    print("  [SUCCESS] ALL TESTS PASSED")
    print("=" * 65)
