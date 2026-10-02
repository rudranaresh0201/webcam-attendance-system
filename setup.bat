@echo off
rem One-time setup on Windows: creates .venv, installs packages, checks the install.
cd /d "%~dp0"

python -m venv .venv || goto :error
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip || goto :error
pip install -r requirements.txt || goto :error
python download_models.py || goto :error
python check_setup.py --no-camera || goto :error

echo.
echo Setup done. Activate the environment with:  .venv\Scripts\activate
exit /b 0

:error
echo.
echo Setup failed. See the message above.
exit /b 1
