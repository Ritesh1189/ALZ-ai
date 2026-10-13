@echo off
title ALZ ai - AI Resume Analyzer
echo ========================================================
echo   Starting ALZ ai Resume Analyzer
echo   Localhost URL: http://localhost:8501
echo ========================================================
cd /d "%~dp0"
if exist "venv\Scripts\python.exe" (
    venv\Scripts\python.exe -m streamlit run app.py --server.port 8501 --server.address localhost
) else (
    python -m streamlit run app.py --server.port 8501 --server.address localhost
)
pause
