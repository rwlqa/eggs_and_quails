#!/bin/bash
echo "======================================================"
echo "  TomatoApp: Установка библиотек и запуск..."
echo "======================================================"

# Проверка Python
if ! command -v python3 &> /dev/null
then
    echo "[ERROR] Python3 не найден! Установите его через apt/brew."
    exit
fi

echo "[1/2] Установка зависимостей..."
pip3 install -r requirements.txt

echo "[2/2] Запуск..."
python3 main.py
