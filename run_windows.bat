@echo off
echo ======================================================
echo   TomatoApp: Установка библиотек и запуск...
echo ======================================================

:: Проверка наличия Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python не найден! Пожалуйста, установите Python 3.10+
    pause
    exit /b
)

echo [1/2] Установка зависимостей (это может занять время)...
pip install -r requirements.txt

echo [2/2] Запуск приложения...
python main.py

if %errorlevel% neq 0 (
    echo [ERROR] Приложение завершилось с ошибкой.
    pause
)
