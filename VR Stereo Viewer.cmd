@echo off
chcp 65001 >nul

:: Проверяем наличие виртуального окружения
if exist "venv\Scripts\pythonw.exe" (
    start "" "venv\Scripts\pythonw.exe" "launcher.pyw"
) else (
    start "" pythonw "launcher.pyw"
)

