@echo off
chcp 65001 >nul
title VR Stereo Image Viewer

echo ═══════════════════════════════════════════════════
echo         VR Stereo Image Viewer
echo ═══════════════════════════════════════════════════
echo.

:: Проверяем наличие Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ОШИБКА] Python не найден!
    echo Установите Python с https://python.org
    pause
    exit /b 1
)

:: Проверяем наличие виртуального окружения
if exist "venv\Scripts\activate.bat" (
    echo Активация виртуального окружения...
    call venv\Scripts\activate.bat
)

:: Проверяем установлены ли зависимости
python -c "import xr" >nul 2>&1
if errorlevel 1 (
    echo Установка зависимостей...
    pip install -r requirements.txt
    echo.
)

:: Запускаем приложение
echo Запуск VR Stereo Viewer...
echo Лог сохраняется в: vr_viewer.log
echo.
python sbs_viewer.py %*

:: Показываем результат
echo.
if errorlevel 1 (
    echo [ОШИБКА] Приложение завершилось с ошибкой
    echo.
    echo Последние строки лога:
    echo ─────────────────────────────────────────────────
    powershell -Command "Get-Content vr_viewer.log -Tail 20"
    echo ─────────────────────────────────────────────────
) else (
    echo Приложение завершено успешно
)
echo.
echo Полный лог: vr_viewer.log
pause

