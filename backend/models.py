from datetime import datetime
from sqlalchemy import (
    Column, Integer, String, Float, Boolean, DateTime, ForeignKey, Text, Enum
)
from sqlalchemy.orm import relationship
import enum
from backend.database import Base


class UserRole(str, enum.Enum):
    ADMIN = "ADMIN"
    HOD = "HOD"
    STAFF = "STAFF"
    CLASS = "CLASS"


class AttendanceStatus(str, enum.Enum):
    PRESENT = "PRESENT"
    ABSENT = "ABSENT"
    REVIEW = "REVIEW"


class SessionStatus(str, enum.Enum):
    PENDING_REVIEW = "PENDING_REVIEW"
    CONFIRMED = "CONFIRMED"


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    full_name = Column(String(100), nullable=False)
    role = Column(String(20), nullable=False, default=UserRole.STAFF.value)
    department_id = Column(Integer, ForeignKey("departments.id"), nullable=True)

    department = relationship("Department", back_populates="users")
    assignments = relationship("StaffClassAssignment", back_populates="staff_user")
    sessions_taken = relationship("AttendanceSession", back_populates="taken_by_user", foreign_keys="[AttendanceSession.taken_by_user_id]")


class Department(Base):
    __tablename__ = "departments"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), unique=True, nullable=False)
    code = Column(String(20), unique=True, nullable=False)

    users = relationship("User", back_populates="department")
    classes = relationship("ClassRoom", back_populates="department")


class ClassRoom(Base):
    __tablename__ = "classes"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)  # e.g. "III AIDS A"
    department_id = Column(Integer, ForeignKey("departments.id"), nullable=False)

    department = relationship("Department", back_populates="classes")
    students = relationship("Student", back_populates="class_room")
    subjects = relationship("Subject", back_populates="class_room")
    staff_assignments = relationship("StaffClassAssignment", back_populates="class_room")
    sessions = relationship("AttendanceSession", back_populates="class_room")


class Subject(Base):
    __tablename__ = "subjects"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)  # e.g. "Artificial Intelligence"
    code = Column(String(20), nullable=False)   # e.g. "AI301"
    class_id = Column(Integer, ForeignKey("classes.id"), nullable=False)

    class_room = relationship("ClassRoom", back_populates="subjects")
    sessions = relationship("AttendanceSession", back_populates="subject")


class Student(Base):
    __tablename__ = "students"

    id = Column(Integer, primary_key=True, index=True)
    student_id = Column(String(50), unique=True, index=True, nullable=False)  # Reg No e.g. "922524243005"
    name = Column(String(100), nullable=False)
    class_id = Column(Integer, ForeignKey("classes.id"), nullable=False)
    photo_path = Column(String(255), nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    class_room = relationship("ClassRoom", back_populates="students")
    embeddings = relationship("StudentFaceEmbedding", back_populates="student", cascade="all, delete-orphan")
    attendance_records = relationship("AttendanceRecord", back_populates="student")


class StudentFaceEmbedding(Base):
    __tablename__ = "student_face_embeddings"

    id = Column(Integer, primary_key=True, index=True)
    student_id = Column(Integer, ForeignKey("students.id"), nullable=False)
    embedding_data = Column(Text, nullable=False)  # JSON array of 128 floats
    quality_score = Column(Float, default=1.0)
    created_at = Column(DateTime, default=datetime.utcnow)

    student = relationship("Student", back_populates="embeddings")


class StaffClassAssignment(Base):
    __tablename__ = "staff_class_assignments"

    id = Column(Integer, primary_key=True, index=True)
    staff_user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    class_id = Column(Integer, ForeignKey("classes.id"), nullable=False)
    subject_id = Column(Integer, ForeignKey("subjects.id"), nullable=True)

    staff_user = relationship("User", back_populates="assignments")
    class_room = relationship("ClassRoom", back_populates="staff_assignments")


class AttendanceSession(Base):
    __tablename__ = "attendance_sessions"

    id = Column(Integer, primary_key=True, index=True)
    class_id = Column(Integer, ForeignKey("classes.id"), nullable=False)
    subject_id = Column(Integer, ForeignKey("subjects.id"), nullable=True)
    taken_by_user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    session_date = Column(String(20), nullable=False)  # YYYY-MM-DD
    session_time = Column(String(20), nullable=False)  # HH:MM:SS
    status = Column(String(30), default=SessionStatus.PENDING_REVIEW.value)
    total_students = Column(Integer, default=0)
    present_count = Column(Integer, default=0)
    absent_count = Column(Integer, default=0)
    pending_count = Column(Integer, default=0)
    confirmed_by_user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    confirmed_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    class_room = relationship("ClassRoom", back_populates="sessions")
    subject = relationship("Subject", back_populates="sessions")
    taken_by_user = relationship("User", back_populates="sessions_taken", foreign_keys=[taken_by_user_id])
    confirmed_by_user = relationship("User", foreign_keys=[confirmed_by_user_id])
    records = relationship("AttendanceRecord", back_populates="session", cascade="all, delete-orphan")
    reviews = relationship("AttendanceReview", back_populates="session", cascade="all, delete-orphan")


class AttendanceRecord(Base):
    __tablename__ = "attendance_records"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("attendance_sessions.id"), nullable=False)
    student_id = Column(Integer, ForeignKey("students.id"), nullable=False)
    status = Column(String(20), nullable=False, default=AttendanceStatus.ABSENT.value)
    confidence = Column(Float, default=0.0)
    confirmed_by_user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    confirmation_time = Column(DateTime, nullable=True)
    notes = Column(String(255), nullable=True)

    session = relationship("AttendanceSession", back_populates="records")
    student = relationship("Student", back_populates="attendance_records")


class AttendanceReview(Base):
    __tablename__ = "attendance_reviews"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("attendance_sessions.id"), nullable=False)
    student_id = Column(Integer, ForeignKey("students.id"), nullable=False)
    captured_face_crop = Column(String(255), nullable=True)  # File path to cropped webcam face
    match_score = Column(Float, default=0.0)
    review_status = Column(String(20), default="PENDING")  # PENDING, APPROVED_PRESENT, MARKED_ABSENT

    session = relationship("AttendanceSession", back_populates="reviews")
    student = relationship("Student")


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    username = Column(String(50), nullable=True)
    action = Column(String(100), nullable=False)
    details = Column(Text, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
