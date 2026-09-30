@echo off
title VSB AI Attendance Portal
echo ========================================================
echo Starting VSB AI Automated Attendance Portal locally...
echo URL: http://127.0.0.1:8000
echo API Docs: http://127.0.0.1:8000/docs
echo ========================================================
python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
pause
