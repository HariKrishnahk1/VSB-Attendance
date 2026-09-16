import os
import shutil
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, status
from sqlalchemy.orm import Session
from backend.database import get_db
from backend.models import (
    User, UserRole, Department, ClassRoom, Subject, Student, AuditLog
)
from backend.schemas import (
    UserCreate, UserResponse, DepartmentCreate, ClassCreate, SubjectCreate, ThresholdConfig
)
from backend.services.auth_service import get_password_hash, require_roles
from backend.services.vision_service import process_zip_dataset
from backend.config import UPLOAD_DIR, THRESHOLD_HIGH_CONFIDENCE, THRESHOLD_MEDIUM_CONFIDENCE

router = APIRouter(prefix="/api/admin", tags=["Admin Management"])


@router.post("/departments")
def create_department(
    req: DepartmentCreate,
    db: Session = Depends(get_db),
    admin: User = Depends(require_roles([UserRole.ADMIN]))
):
    existing = db.query(Department).filter(Department.code == req.code).first()
    if existing:
        raise HTTPException(status_code=400, detail="Department code already exists.")

    dept = Department(name=req.name, code=req.code)
    db.query
    db.add(dept)
    db.commit()
    db.refresh(dept)
    return dept


@router.get("/departments")
def get_departments(db: Session = Depends(get_db)):
    return db.query(Department).all()


@router.post("/classes")
def create_class(
    req: ClassCreate,
    db: Session = Depends(get_db),
    admin: User = Depends(require_roles([UserRole.ADMIN]))
):
    c = ClassRoom(name=req.name, department_id=req.department_id)
    db.add(c)
    db.commit()
    db.refresh(c)
    return c


@router.get("/classes")
def get_classes(db: Session = Depends(get_db)):
    classes = db.query(ClassRoom).all()
    res = []
    for c in classes:
        res.append({
            "id": c.id,
            "name": c.name,
            "department_id": c.department_id,
            "department_name": c.department.name if c.department else "",
            "student_count": len(c.students)
        })
    return res


@router.post("/subjects")
def create_subject(
    req: SubjectCreate,
    db: Session = Depends(get_db),
    admin: User = Depends(require_roles([UserRole.ADMIN]))
):
    subj = Subject(name=req.name, code=req.code, class_id=req.class_id)
    db.add(subj)
    db.commit()
    db.refresh(subj)
    return subj


@router.get("/subjects")
def get_subjects(class_id: int = None, db: Session = Depends(get_db)):
    query = db.query(Subject)
    if class_id:
        query = query.filter(Subject.class_id == class_id)
    return query.all()


@router.post("/users", response_model=UserResponse)
def create_user(
    req: UserCreate,
    db: Session = Depends(get_db),
    admin: User = Depends(require_roles([UserRole.ADMIN]))
):
    existing = db.query(User).filter(User.username == req.username).first()
    if existing:
        raise HTTPException(status_code=400, detail="Username already exists.")

    user = User(
        username=req.username,
        password_hash=get_password_hash(req.password),
        full_name=req.full_name,
        role=req.role,
        department_id=req.department_id
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@router.get("/users")
def get_users(
    db: Session = Depends(get_db),
    admin: User = Depends(require_roles([UserRole.ADMIN]))
):
    users = db.query(User).all()
    return [{
        "id": u.id,
        "username": u.username,
        "full_name": u.full_name,
        "role": u.role,
        "department_id": u.department_id,
        "department_name": u.department.name if u.department else None
    } for u in users]


@router.post("/upload-zip-dataset")
async def upload_zip_dataset(
    class_id: int = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    admin: User = Depends(require_roles([UserRole.ADMIN]))
):
    if not file.filename.endswith(".zip"):
        raise HTTPException(status_code=400, detail="File must be a .zip archive.")

    temp_zip_path = UPLOAD_DIR / f"upload_{file.filename}"
    with open(temp_zip_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        summary = process_zip_dataset(str(temp_zip_path), class_id, db)
        
        # Log Audit
        log = AuditLog(
            user_id=admin.id,
            username=admin.username,
            action="ZIP_DATASET_UPLOAD",
            details=f"Uploaded {file.filename} for Class ID {class_id}. Processed {summary['successfully_processed']}/{summary['total_students']} students."
        )
        db.add(log)
        db.commit()
        return summary
    finally:
        if temp_zip_path.exists():
            os.remove(temp_zip_path)


@router.get("/students")
def get_students(class_id: int = None, db: Session = Depends(get_db)):
    query = db.query(Student)
    if class_id:
        query = query.filter(Student.class_id == class_id)
    students = query.all()
    return [{
        "id": s.id,
        "student_id": s.student_id,
        "name": s.name,
        "class_id": s.class_id,
        "class_name": s.class_room.name if s.class_room else "N/A",
        "photo_path": s.photo_path,
        "has_embedding": len(s.embeddings) > 0
    } for s in students]


@router.get("/audit-logs")
def get_audit_logs(
    db: Session = Depends(get_db),
    admin: User = Depends(require_roles([UserRole.ADMIN]))
):
    logs = db.query(AuditLog).order_by(AuditLog.timestamp.desc()).limit(100).all()
    return logs
