@echo off
title Options Hunter — First Time Setup
echo.
echo  ============================================
echo   OPTIONS HUNTER — Installing Requirements
echo  ============================================
echo.
echo  This only needs to run ONCE.
echo  Installing required Python packages...
echo.

python -m pip install flask yfinance numpy scipy pandas --upgrade

echo.
echo  ============================================
echo   Setup complete! You can close this window.
echo   From now on just double-click: run.bat
echo  ============================================
echo.
pause
