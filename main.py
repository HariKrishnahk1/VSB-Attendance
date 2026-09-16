import os
import shutil
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from backend.config import BASE_DIR, UPLOAD_DIR, STUDENT_PHOTOS_DIR
from backend.database import engine, Base, SessionLocal
from backend.models import (
    User, UserRole, Department, ClassRoom, Subject, Student
)
from backend.services.auth_service import get_password_hash
from backend.services.vision_service import process_zip_dataset, get_vision_engine
from backend.routers import auth, admin, attendance, hod

# Initialize FastAPI App
app = FastAPI(
    title="AI Automated Face-Recognition Attendance Portal",
    description="Production Classroom Smartboard Attendance System",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Explicit Static Handlers
@app.get("/static/css/styles.css")
@app.get("/css/styles.css")
@app.get("/styles.css")
def get_css():
    css_path = BASE_DIR / "static" / "css" / "styles.css"
    if css_path.exists():
        return FileResponse(str(css_path), media_type="text/css")
    return Response(status_code=404)

@app.get("/static/js/app.js")
@app.get("/js/app.js")
@app.get("/app.js")
def get_js():
    js_path = BASE_DIR / "static" / "js" / "app.js"
    if js_path.exists():
        return FileResponse(str(js_path), media_type="application/javascript")
    return Response(status_code=404)

@app.get("/favicon.ico")
@app.get("/favicon.png")
@app.get("/logo.png")
def get_favicon():
    logo_path = BASE_DIR / "static" / "img" / "logo.png"
    if logo_path.exists():
        return FileResponse(str(logo_path), media_type="image/png")
    return Response(status_code=404)

# Mount Static Files & Uploads
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
app.mount("/static/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")

# Include Routers
app.include_router(auth.router)
app.include_router(admin.router)
app.include_router(attendance.router)
app.include_router(hod.router)


def seed_database():
    """Initializes tables, default accounts, and auto-ingests student ZIP dataset if present."""
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        # 1. Department
        dept = db.query(Department).filter(Department.code == "AIDS").first()
        if not dept:
            dept = Department(name="Artificial Intelligence & Data Science", code="AIDS")
            db.add(dept)
            db.commit()
            db.refresh(dept)

        # 2. Class
        c_aids = db.query(ClassRoom).filter(ClassRoom.name == "III AIDS A").first()
        if not c_aids:
            c_aids = ClassRoom(name="III AIDS A", department_id=dept.id)
            db.add(c_aids)
            db.commit()
            db.refresh(c_aids)

        # 3. Subject
        subj_ai = db.query(Subject).filter(Subject.code == "AI301").first()
        if not subj_ai:
            subj_ai = Subject(name="Artificial Intelligence", code="AI301", class_id=c_aids.id)
            db.add(subj_ai)
            subj_ml = Subject(name="Machine Learning", code="ML302", class_id=c_aids.id)
            db.add(subj_ml)
            db.commit()

        # 4. Default Accounts
        # Admin
        if not db.query(User).filter(User.username == "admin").first():
            db.add(User(
                username="admin",
                password_hash=get_password_hash("admin123"),
                full_name="System Administrator",
                role=UserRole.ADMIN.value
            ))

        # HOD
        if not db.query(User).filter(User.username == "hod_aids").first():
            db.add(User(
                username="hod_aids",
                password_hash=get_password_hash("hod123"),
                full_name="Dr. Aris (HOD AI & DS)",
                role=UserRole.HOD.value,
                department_id=dept.id
            ))

        # Staff
        if not db.query(User).filter(User.username == "staff_hari").first():
            db.add(User(
                username="staff_hari",
                password_hash=get_password_hash("staff123"),
                full_name="Prof. Hari (Faculty)",
                role=UserRole.STAFF.value,
                department_id=dept.id
            ))

        # Class Smartboard
        if not db.query(User).filter(User.username == "class_3aids_a").first():
            db.add(User(
                username="class_3aids_a",
                password_hash=get_password_hash("class123"),
                full_name="Smartboard III AIDS A",
                role=UserRole.CLASS.value,
                department_id=dept.id
            ))

        db.commit()

        # 5. Check and auto-ingest "I Year AIDS A Sec Students Photo.zip" if present and students table is empty
        zip_candidates = list(BASE_DIR.glob("*.zip"))
        if zip_candidates and db.query(Student).count() == 0:
            zip_path = zip_candidates[0]
            print(f"[Startup] Found initial student ZIP dataset: {zip_path.name}. Auto-processing...")
            try:
                summary = process_zip_dataset(str(zip_path), c_aids.id, db)
                print(f"[Startup] Dataset Ingestion Complete! Processed {summary['successfully_processed']} / {summary['total_students']} students.")
            except Exception as e:
                print(f"[Startup] Error ingesting initial ZIP dataset: {e}")

    finally:
        db.close()


# Synchronously ensure database and default accounts exist on serverless cold start
try:
    seed_database()
except Exception as e:
    print(f"[Auto-Init] Database seed exception: {e}")


@app.on_event("startup")
def startup_event():
    print("[Startup] App started. Database ready.")


@app.get("/")
@app.get("/index.html")
def read_root():
    index_path = BASE_DIR / "static" / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"message": "AI Automated Attendance Portal API is running", "docs": "/docs"}
