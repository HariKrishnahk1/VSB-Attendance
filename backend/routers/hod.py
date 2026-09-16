from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from datetime import datetime
from backend.database import get_db
from backend.models import User, UserRole, Department, ClassRoom, AttendanceSession, Student
from backend.services.auth_service import require_roles

router = APIRouter(prefix="/api/hod", tags=["HOD Analytics"])


@router.get("/dashboard")
def get_hod_dashboard(
    date: str = None,
    db: Session = Depends(get_db),
    hod: User = Depends(require_roles([UserRole.HOD, UserRole.ADMIN]))
):
    dept_id = hod.department_id
    if not dept_id and hod.role == UserRole.ADMIN:
        dept = db.query(Department).first()
        dept_id = dept.id if dept else None

    if not dept_id:
        raise HTTPException(status_code=400, detail="HOD is not assigned to a department.")

    dept = db.query(Department).filter(Department.id == dept_id).first()
    classes = db.query(ClassRoom).filter(ClassRoom.department_id == dept_id).all()

    target_date = date or datetime.utcnow().strftime("%Y-%m-%d")

    class_stats = []
    dept_total_students = 0
    dept_present = 0
    dept_absent = 0

    for c in classes:
        st_count = len(c.students)
        dept_total_students += st_count

        # Get latest attendance session for target_date
        session = db.query(AttendanceSession).filter(
            AttendanceSession.class_id == c.id,
            AttendanceSession.session_date == target_date
        ).order_by(AttendanceSession.created_at.desc()).first()

        if session:
            present = session.present_count
            absent = session.absent_count
            total = session.total_students or 1
            pct = round((present / total) * 100, 2)
            st_status = session.status
        else:
            present = 0
            absent = 0
            pct = 0.0
            st_status = "NO_DATA"

        dept_present += present
        dept_absent += absent

        class_stats.append({
            "class_id": c.id,
            "class_name": c.name,
            "total_students": st_count,
            "present": present,
            "absent": absent,
            "percentage": f"{pct:.2f}%",
            "status": st_status
        })

    overall_pct = round((dept_present / dept_total_students * 100), 2) if dept_total_students > 0 else 0.0

    return {
        "department_name": dept.name if dept else "AI & DS",
        "department_code": dept.code if dept else "AIDS",
        "date": target_date,
        "total_classes": len(classes),
        "total_students": dept_total_students,
        "dept_present": dept_present,
        "dept_absent": dept_absent,
        "overall_percentage": f"{overall_pct:.2f}%",
        "classes": class_stats
    }
