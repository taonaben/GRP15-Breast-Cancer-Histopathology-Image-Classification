@echo off
echo Starting Breast Cancer Classification System...
echo.

echo Starting FastAPI server...
start cmd /k "cd /d %~dp0 && python -m uvicorn server.app:app --reload --host 0.0.0.0 --port 8000"

timeout /t 3 /nobreak > nul

echo Starting Streamlit frontend...
start cmd /k "cd /d %~dp0 && streamlit run frontend/main.py"

echo.
echo Both servers are starting...
echo FastAPI: http://localhost:8000
echo Streamlit: http://localhost:8501
echo.
pause
