@echo off
set pd=%~dp0
set LABEL_STUDIO_WORKSPACE=%pd%..\workspace\
start python %pd%..\model\backend\_wsgi.py
REM label-studio-ml start %pd%..\workspace\model_v0
start python %pd%..\annotation\label-studio\label_studio\manage.py runserver --noreload
REM start label-studio start
