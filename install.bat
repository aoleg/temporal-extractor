@echo off
setlocal
cd /d "%~dp0"

echo temporal_extractor setup
echo.

REM No specific Python version is required -- the tool uses nothing exotic.
REM Whatever "python" resolves to is what the venv will be built from.
where python >nul 2>&1
if errorlevel 1 (
    echo ERROR: 'python' was not found on PATH.
    echo Install Python and make sure it is on PATH, then run this again.
    exit /b 1
)

for /f "delims=" %%v in ('python -c "import sys;print(sys.version.split()[0])"') do set PYVER=%%v
echo Using Python %PYVER%

if exist ".venv\Scripts\python.exe" (
    echo Virtualenv already exists, reusing it.
) else (
    echo Creating virtualenv in .venv ...
    python -m venv .venv
    if errorlevel 1 (
        echo ERROR: could not create the virtualenv.
        exit /b 1
    )
)

echo Installing dependencies ...
".venv\Scripts\python.exe" -m pip install --upgrade pip --quiet
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: dependency installation failed.
    exit /b 1
)

if not exist ".env" (
    copy ".env.example" ".env" >nul
    echo.
    echo Created .env from the template.
)

echo.
echo ------------------------------------------------------------
echo Setup complete.
echo.
echo NEXT STEP: open .env and set SEEDVR2_REPO to your SeedVR2
echo checkout. The restorer also needs its own separate virtualenv
echo with torch installed -- see README.md.
echo.
echo Then check everything is wired up:
echo     extract.bat doctor
echo ------------------------------------------------------------
endlocal
