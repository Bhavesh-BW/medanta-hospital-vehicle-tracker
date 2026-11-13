@echo off
title Medanta Hospital - Vehicle Access Control & ANPR System
echo ======================================================================
echo   Medanta Hospital - Vehicle Access Control & ANPR System
echo   Opening at: http://localhost:8176
echo ======================================================================
cd /d "%~dp0backend"
python vehicle_tracker.py
pause
