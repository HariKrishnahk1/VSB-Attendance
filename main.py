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
    """
    Initializes all DB tables and seeds default department, class, and user accounts.
    Each step is independently error-handled with rollback so a single failure
    does not leave the DB session in a broken state.
    """
    print("[DB Init] Running database initialization...")
    try:
        Base.metadata.create_all(bind=engine)
        print("[DB Init] All tables created/verified OK.")
    except Exception as e:
        print(f"[DB Init] CRITICAL: Could not create tables: {e}")
        raise

    db = SessionLocal()
    try:
        # Step 1: Department
        try:
            dept = db.query(Department).filter(Department.code == "AIDS").first()
            if not dept:
                dept = Department(name="Artificial Intelligence & Data Science", code="AIDS")
                db.add(dept)
                db.commit()
                db.refresh(dept)
                print("[DB Init] Department 'AIDS' created.")
            else:
                print(f"[DB Init] Department exists: {dept.name}")
        except Exception as e:
            db.rollback()
            print(f"[DB Init] ERROR in department seed: {e}")
            raise

        # Step 2: Classroom
        try:
            c_aids = db.query(ClassRoom).filter(ClassRoom.name == "III AIDS A").first()
            if not c_aids:
                c_aids = ClassRoom(name="III AIDS A", department_id=dept.id)
                db.add(c_aids)
                db.commit()
                db.refresh(c_aids)
                print("[DB Init] ClassRoom 'III AIDS A' created.")
            else:
                print(f"[DB Init] ClassRoom exists: {c_aids.name}")
        except Exception as e:
            db.rollback()
            print(f"[DB Init] ERROR in classroom seed: {e}")
            raise

        # Step 3: Default User Accounts
        try:
            default_users = [
                {"username": "admin",       "password": "admin123", "full_name": "System Administrator",      "role": UserRole.ADMIN.value,  "dept": False},
                {"username": "hod_aids",    "password": "hod123",   "full_name": "Dr. Aris (HOD AI & DS)",    "role": UserRole.HOD.value,   "dept": True},
                {"username": "staff_hari",  "password": "staff123", "full_name": "Prof. Hari (Faculty)",      "role": UserRole.STAFF.value,  "dept": True},
                {"username": "class_3aids_a","password": "class123","full_name": "Smartboard III AIDS A",      "role": UserRole.CLASS.value,  "dept": True},
            ]
            for u in default_users:
                if not db.query(User).filter(User.username == u["username"]).first():
                    db.add(User(
                        username=u["username"],
                        password_hash=get_password_hash(u["password"]),
                        full_name=u["full_name"],
                        role=u["role"],
                        department_id=dept.id if u["dept"] else None
                    ))
                    print(f"[DB Init] User '{u['username']}' ({u['role']}) created.")
            db.commit()
            print("[DB Init] All default accounts verified/created OK.")
        except Exception as e:
            db.rollback()
            print(f"[DB Init] ERROR in user accounts seed: {e}")
            raise

        # Step 4: Auto-ingest student ZIP dataset on fresh deployment
        try:
            zip_candidates = list(BASE_DIR.glob("*.zip"))
            if zip_candidates and db.query(Student).count() == 0:
                zip_path = zip_candidates[0]
                print(f"[DB Init] Auto-ingesting student dataset: {zip_path.name}...")
                summary = process_zip_dataset(str(zip_path), c_aids.id, db)
                print(f"[DB Init] Dataset ingestion complete: {summary['successfully_processed']}/{summary['total_students']} students processed.")
            else:
                student_count = db.query(Student).count()
                print(f"[DB Init] Student table has {student_count} records — skipping ZIP auto-ingest.")
        except Exception as e:
            print(f"[DB Init] WARNING: ZIP auto-ingest failed (non-critical): {e}")

        print("[DB Init] Database initialization complete.")

    finally:
        db.close()


# Synchronously ensure database and default accounts exist on cold start
try:
    seed_database()
except Exception as e:
    print(f"[Auto-Init] FATAL: Database seed failed — {e}. App may be degraded.")


@app.on_event("startup")
def startup_event():
    print("[Startup] App started. Database ready.")


@app.get("/api/health")
def health_check():
    """Health check endpoint — reports database connectivity, table counts, and system status."""
    from sqlalchemy import text
    status = {"status": "ok", "database": {}, "errors": []}
    db = SessionLocal()
    try:
        db.execute(text("SELECT 1"))
        status["database"]["connection"] = "ok"
        status["database"]["students"]   = db.query(Student).count()
        status["database"]["users"]       = db.query(User).count()
        status["database"]["sessions"]    = db.execute(text("SELECT COUNT(*) FROM attendance_sessions")).scalar()
        status["database"]["embeddings"]  = db.execute(text("SELECT COUNT(*) FROM student_face_embeddings")).scalar()
    except Exception as e:
        status["status"] = "error"
        status["database"]["connection"] = "FAILED"
        status["errors"].append(str(e))
    finally:
        db.close()
    return status


@app.post("/api/db-reset")
def db_reset():
    """Emergency endpoint to re-run database seed on Render cold-start or after schema migration."""
    try:
        seed_database()
        db = SessionLocal()
        student_count = db.query(Student).count()
        user_count = db.query(User).count()
        db.close()
        return {
            "status": "success",
            "message": "Database re-initialized successfully.",
            "students": student_count,
            "users": user_count
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/")
@app.get("/index.html")
def read_root():
    index_path = BASE_DIR / "static" / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"message": "AI Automated Attendance Portal API is running", "docs": "/docs"}
