@echo off
set pd=%~dp0
python %pd%..\annotation\label-studio\label_studio\service.py install
python %pd%..\model\backend\service.py install
pause
