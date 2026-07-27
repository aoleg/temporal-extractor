@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo ERROR: no virtualenv found. Run install.bat first.
    exit /b 1
)

REM The reference SeedVR2 code prints emoji at import; under a cp1252 console
REM that is a UnicodeEncodeError before any work happens.
set PYTHONIOENCODING=utf-8

".venv\Scripts\python.exe" "%~dp0extract.py" %*
exit /b %errorlevel%
