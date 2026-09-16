from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: str
    username: str
    full_name: str
    department_id: Optional[int] = None


class LoginRequest(BaseModel):
    username: str
    password: str


class UserCreate(BaseModel):
    username: str
    password: str
    full_name: str
    role: str
    department_id: Optional[int] = None


class UserResponse(BaseModel):
    id: int
    username: str
    full_name: str
    role: str
    department_id: Optional[int] = None

    class Config:
        from_attributes = True


class DepartmentCreate(BaseModel):
    name: str
    code: str


class ClassCreate(BaseModel):
    name: str
    department_id: int


class SubjectCreate(BaseModel):
    name: str
    code: str
    class_id: int


class FrameProcessRequest(BaseModel):
    class_id: int
    subject_id: Optional[int] = None
    frames: List[str]  # Base64 encoded JPEG strings captured from webcam


class AttendanceConfirmItem(BaseModel):
    student_id: int
    status: str  # PRESENT or ABSENT


class AttendanceConfirmRequest(BaseModel):
    session_id: int
    confirmations: List[AttendanceConfirmItem]


class ThresholdConfig(BaseModel):
    high_threshold: float
    medium_threshold: float
