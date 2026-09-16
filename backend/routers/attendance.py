import os
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from backend.database import get_db
from backend.models import (
    User, UserRole, AttendanceSession, AttendanceRecord, AttendanceReview, Student, ClassRoom, Subject
)
from backend.schemas import FrameProcessRequest, AttendanceConfirmRequest
from backend.services.auth_service import get_current_user, require_roles
from backend.services.attendance_service import process_smartboard_session, confirm_staff_reviews
from backend.services.excel_service import generate_attendance_excel
from backend.config import EXCEL_OUTPUT_DIR

router = APIRouter(prefix="/api/attendance", tags=["Attendance Management"])


@router.post("/process-frames")
def process_frames(
    req: FrameProcessRequest,
    db: Session = Depends(get_db),
    user: User = Depends(require_roles([UserRole.CLASS, UserRole.STAFF, UserRole.ADMIN]))
):
    if not req.frames or len(req.frames) == 0:
        raise HTTPException(status_code=400, detail="No webcam frames received.")

    try:
        res = process_smartboard_session(
            class_id=req.class_id,
            subject_id=req.subject_id,
            taken_by_user_id=user.id,
            base64_frames=req.frames,
            db_session=db
        )
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/session/{session_id}")
def get_session_details(
    session_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user)
):
    session = db.query(AttendanceSession).filter(AttendanceSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")

    records = db.query(AttendanceRecord).filter(AttendanceRecord.session_id == session_id).all()
    reviews = db.query(AttendanceReview).filter(AttendanceReview.session_id == session_id).all()

    record_items = []
    for r in records:
        record_items.append({
            "id": r.id,
            "student_id": r.student.student_id if r.student else "",
            "student_db_id": r.student.id if r.student else None,
            "student_name": r.student.name if r.student else "",
            "registered_photo": r.student.photo_path if r.student else "",
            "status": r.status,
            "confidence": r.confidence,
            "notes": r.notes
        })

    review_items = []
    for rv in reviews:
        review_items.append({
            "id": rv.id,
            "student_id": rv.student.student_id if rv.student else "",
            "student_db_id": rv.student_id,
            "student_name": rv.student.name if rv.student else "",
            "captured_crop": rv.captured_face_crop,
            "registered_photo": rv.student.photo_path if rv.student else "",
            "match_score": round(rv.match_score * 100, 1),
            "review_status": rv.review_status
        })

    return {
        "session_id": session.id,
        "class_id": session.class_id,
        "class_name": session.class_room.name if session.class_room else "",
        "subject_name": session.subject.name if session.subject else "",
        "date": session.session_date,
        "time": session.session_time,
        "status": session.status,
        "total_students": session.total_students,
        "present_count": session.present_count,
        "absent_count": session.absent_count,
        "pending_count": session.pending_count,
        "records": record_items,
        "reviews": review_items
    }


@router.post("/confirm-review")
def confirm_review(
    req: AttendanceConfirmRequest,
    db: Session = Depends(get_db),
    user: User = Depends(require_roles([UserRole.STAFF, UserRole.ADMIN]))
):
    confirmations_data = [item.dict() for item in req.confirmations]
    updated_session = confirm_staff_reviews(
        session_id=req.session_id,
        confirmations=confirmations_data,
        confirmed_by_user_id=user.id,
        db_session=db
    )

    return {
        "message": "Staff confirmation saved successfully.",
        "session_id": updated_session.id,
        "status": updated_session.status,
        "present_count": updated_session.present_count,
        "absent_count": updated_session.absent_count,
        "pending_count": updated_session.pending_count
    }


@router.post("/generate-excel/{session_id}")
def generate_excel(
    session_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(require_roles([UserRole.STAFF, UserRole.HOD, UserRole.ADMIN]))
):
    session = db.query(AttendanceSession).filter(AttendanceSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")

    if session.pending_count > 0:
        raise HTTPException(
            status_code=400,
            detail="Cannot generate final Excel sheet until all uncertain students are confirmed by staff."
        )

    file_name = generate_attendance_excel(session_id, db)
    return {"file_name": file_name, "download_url": f"/api/attendance/download-excel/{file_name}"}


@router.get("/download-excel/{file_name}")
def download_excel(
    file_name: str,
    user: User = Depends(get_current_user)
):
    file_path = EXCEL_OUTPUT_DIR / file_name
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Excel file not found.")

    return FileResponse(
        path=file_path,
        filename=file_name,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )


@router.get("/history")
def get_attendance_history(
    class_id: int = None,
    subject_id: int = None,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user)
):
    query = db.query(AttendanceSession)

    # Restrict staff/class logins to assigned class
    if user.role == UserRole.STAFF:
        # Check staff class assignments
        assigned_class_ids = [a.class_id for a in user.assignments]
        if assigned_class_ids:
            query = query.filter(AttendanceSession.class_id.in_(assigned_class_ids))
    elif user.role == UserRole.CLASS:
        # Smartboard login
        pass
    elif user.role == UserRole.HOD:
        # Restrict to HOD's department
        if user.department_id:
            query = query.join(ClassRoom).filter(ClassRoom.department_id == user.department_id)

    if class_id:
        query = query.filter(AttendanceSession.class_id == class_id)
    if subject_id:
        query = query.filter(AttendanceSession.subject_id == subject_id)

    sessions = query.order_by(AttendanceSession.created_at.desc()).all()

    res = []
    for s in sessions:
        total = s.total_students or 1
        pct = round((s.present_count / total) * 100, 1)
        res.append({
            "id": s.id,
            "class_name": s.class_room.name if s.class_room else "",
            "subject_name": s.subject.name if s.subject else "",
            "date": s.session_date,
            "time": s.session_time,
            "total_students": s.total_students,
            "present": s.present_count,
            "absent": s.absent_count,
            "pending": s.pending_count,
            "percentage": f"{pct}%",
            "status": s.status
        })
    return res
