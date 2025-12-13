@echo off
chcp 65001 >nul
title Установка VR Stereo Image Viewer

echo ═══════════════════════════════════════════════════
echo    Установка VR Stereo Image Viewer
echo ═══════════════════════════════════════════════════
echo.

:: Проверяем наличие Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ОШИБКА] Python не найден!
    echo.
    echo Установите Python 3.8+ с https://python.org
    echo При установке отметьте "Add Python to PATH"
    echo.
    pause
    exit /b 1
)

echo Python найден:
python --version
echo.

:: Создаём виртуальное окружение
if not exist "venv" (
    echo Создание виртуального окружения...
    python -m venv venv
    echo.
)

:: Активируем виртуальное окружение
echo Активация виртуального окружения...
call venv\Scripts\activate.bat
echo.

:: Обновляем pip
echo Обновление pip...
python -m pip install --upgrade pip
echo.

:: Устанавливаем зависимости
echo Установка зависимостей...
pip install -r requirements.txt
echo.

:: Проверяем установку
echo.
echo ═══════════════════════════════════════════════════
echo Проверка установки...
echo ═══════════════════════════════════════════════════
python -c "import xr; import OpenGL; import glfw; import PIL; import tkinterdnd2; print('✓ Все зависимости установлены!')"
if errorlevel 1 (
    echo.
    echo [ОШИБКА] Не все зависимости установлены корректно
    pause
    exit /b 1
)

echo.
echo ═══════════════════════════════════════════════════
echo    Установка завершена!
echo ═══════════════════════════════════════════════════
echo.
echo Для запуска используйте: run.cmd
echo Или: python sbs_viewer.py
echo.
pause

