@echo off
REM ============================================================
REM run.bat – Start the Compliance Copilot on Windows
REM ============================================================

echo.
echo ╔══════════════════════════════════════════════════════╗
echo ║      GenAI Banking Compliance Copilot v1.0          ║
echo ║      Phi-2 SLM  ·  FAISS  ·  Streamlit             ║
echo ╚══════════════════════════════════════════════════════╝
echo.

REM Copy .env if not exists
IF NOT EXIST ".env" (
    copy .env.example .env
    echo [INFO] Created .env from .env.example
)

REM Install dependencies
echo [STEP 1] Installing dependencies...
pip install -r requirements.txt -q
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] pip install failed. Check your Python environment.
    pause
    exit /b 1
)

REM Pre-ingest sample docs
echo.
echo [STEP 2] Ingesting sample policy documents...
python ingest.py --dir .\data\policies

REM Start backend in new window
echo.
echo [STEP 3] Starting FastAPI backend on port 8000...
start "Compliance Copilot - Backend" cmd /k "cd backend && uvicorn api:app --host 0.0.0.0 --port 8000"
timeout /t 4 /nobreak > NUL

REM Start Streamlit frontend
echo.
echo [STEP 4] Starting Streamlit UI on port 8501...
echo.
echo   Open: http://localhost:8501
echo.

set API_BASE_URL=http://localhost:8000
cd frontend
streamlit run app.py --server.port 8501 --server.address 0.0.0.0 --theme.base dark

pause
