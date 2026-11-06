@echo off
rem Double-click to start the Timelapse Engine GUI (no console window).
cd /d "%~dp0"
start "" pythonw -m timelapse_engine.gui
