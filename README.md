# Eggs and Quails Analyzer

Приложение на Python с графическим интерфейсом (Tkinter) для автоматического анализа изображений. Проект реализует алгоритмы сегментации объектов с использованием нейросети Segment Anything Model (SAM) и их последующей классификации методом Random Forest.

## Функциональные возможности

- Сегментация: Автоматическое выделение помидоров и перепелиных яиц на сложном фоне.
- Классификация: Определение типа объекта (Красный/Оранжевый помидор, Яйцо) и отсев теней.
- OCR: Распознавание букв, составленных из объектов, с помощью библиотеки EasyOCR.
- Визуализация: Поэтапный вывод результатов работы нейросетей в отдельных окнах.

## Установка и настройка

1. Клонируйте репозиторий:
   ```bash
   git clone https://github.com/rwlqa/eggs_and_quails.git
   cd eggs_and_quails

2. Установите необходимые зависимости:

```bash
pip install -r requirements.txt
3. Подготовка моделей:
Создайте папку weights в корне проекта.
Скачайте веса SAM ViT-B по официальной ссылке: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth?spm=a2ty_o01.29997173.0.0.1de155fbdon5nj&file=sam_vit_b_01ec64.pth.
Скачайте модель классификатора tomato_rf_model.pkl по ссылке: https://drive.google.com/file/d/1CvypLtkmQhW7qmaGh5SzWRus5yYHv3dq/view?usp=sharing.
Поместить веса в папку \weights в корневой папке проекта.
