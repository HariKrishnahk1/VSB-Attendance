import sys
import os
import json
import base64
from pathlib import Path
from fastapi.testclient import TestClient

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from main import app
from backend.database import SessionLocal
from backend.models import Student, AttendanceSession, AttendanceRecord, User

client = TestClient(app)


def test_auth():
    print("\n--- 1. Testing Authentication ---")
    roles = [
        ("admin", "admin123", "ADMIN"),
        ("hod_aids", "hod123", "HOD"),
        ("staff_hari", "staff123", "STAFF"),
        ("class_3aids_a", "class123", "CLASS")
    ]
    tokens = {}
    for username, password, expected_role in roles:
        res = client.post("/api/auth/login", json={"username": username, "password": password})
        assert res.status_code == 200, f"Login failed for {username}: {res.text}"
        data = res.json()
        assert data["role"] == expected_role
        tokens[expected_role] = data["access_token"]
        print(f"[OK] Login successful for {username} ({expected_role})")
    return tokens


def test_dataset_ingestion():
    print("\n--- 2. Testing Database Ingestion Metrics ---")
    db = SessionLocal()
    try:
        student_count = db.query(Student).count()
        print(f"[OK] Registered Students in DB: {student_count}")
        assert student_count > 0, "No students found in DB!"
    finally:
        db.close()


def test_smartboard_session(tokens):
    print("\n--- 3. Testing Smartboard Multi-Frame Attendance Capture ---")
    class_headers = {"Authorization": f"Bearer {tokens['CLASS']}"}
    
    # Get classes and subjects
    classes = client.get("/api/admin/classes").json()
    class_id = classes[0]["id"]
    subjects = client.get(f"/api/admin/subjects?class_id={class_id}").json()
    subject_id = subjects[0]["id"]

    # Generate 5 sample frames (1x1 transparent JPEG)
    dummy_b64 = "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAP//////////////////////////////////////////////////////////////////////////////////////wgALCAABAAEBAREA/8QAFBABAAAAAAAAAAAAAAAAAAAAAP/aAAgBAQABPxA="
    frames = [dummy_b64] * 5

    res = client.post(
        "/api/attendance/process-frames",
        json={"class_id": class_id, "subject_id": subject_id, "frames": frames},
        headers=class_headers
    )
    assert res.status_code == 200, f"Process frames failed: {res.text}"
    session_data = res.json()
    print(f"[OK] Smartboard Session created: ID #{session_data['session_id']}")
    print(f"  Total Roster: {session_data['total_students']}, Present: {session_data['present_count']}, Pending: {session_data['pending_count']}, Absent: {session_data['absent_count']}")
    return session_data["session_id"]


def test_staff_review_and_excel(tokens, session_id):
    print("\n--- 4. Testing Staff Confirmation & Excel Generation ---")
    staff_headers = {"Authorization": f"Bearer {tokens['STAFF']}"}

    # Fetch session details
    detail = client.get(f"/api/attendance/session/{session_id}", headers=staff_headers).json()
    print(f"[OK] Fetched Session #{session_id} details. Records count: {len(detail['records'])}")

    # Submit staff confirmation for all records
    confirmations = []
    for r in detail["records"]:
        confirmations.append({
            "student_id": r["student_db_id"],
            "status": "PRESENT" if r["status"] == "PRESENT" else "ABSENT"
        })

    conf_res = client.post(
        "/api/attendance/confirm-review",
        json={"session_id": session_id, "confirmations": confirmations},
        headers=staff_headers
    )
    assert conf_res.status_code == 200, f"Confirmation failed: {conf_res.text}"
    print("[OK] Staff review confirmation saved successfully.")

    # Generate Excel Report
    excel_res = client.post(
        f"/api/attendance/generate-excel/{session_id}",
        headers=staff_headers
    )
    assert excel_res.status_code == 200, f"Excel generation failed: {excel_res.text}"
    excel_info = excel_res.json()
    print(f"[OK] Generated Excel sheet: {excel_info['file_name']}")

    # Download Excel Report
    download_res = client.get(excel_info["download_url"], headers=staff_headers)
    assert download_res.status_code == 200
    assert len(download_res.content) > 1000, "Downloaded Excel file is too small or corrupt!"
    print(f"[OK] Excel download verified! File size: {len(download_res.content)} bytes.")


def test_hod_dashboard(tokens):
    print("\n--- 5. Testing HOD Department Dashboard ---")
    hod_headers = {"Authorization": f"Bearer {tokens['HOD']}"}
    res = client.get("/api/hod/dashboard", headers=hod_headers)
    assert res.status_code == 200
    data = res.json()
    print(f"[OK] HOD Dashboard loaded: Department '{data['department_name']}'")
    print(f"  Total Classes: {data['total_classes']}, Overall Attendance: {data['overall_percentage']}")


if __name__ == "__main__":
    print("Starting System Integration Tests...")
    tokens = test_auth()
    test_dataset_ingestion()
    sid = test_smartboard_session(tokens)
    test_staff_review_and_excel(tokens, sid)
    test_hod_dashboard(tokens)
    print("\n[SUCCESS] ALL VERIFICATION TESTS PASSED SUCCESSFULLY!")
