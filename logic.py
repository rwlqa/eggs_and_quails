import cv2
import numpy as np
import torch
import joblib
import math
import pandas as pd
import easyocr
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from skimage.morphology import skeletonize
import os

FEATURES = [
    'r_med', 'g_med', 'b_med', 'h_med', 's_med', 'v_med', 
    'l_med', 'a_med', 'bl_med', 'rg_ratio', 'rg_diff_bg', 
    'v_ratio_bg', 's_diff_bg', 'std_val', 'edge_energy', 
    'area', 'circularity', 'solidity'
]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SAM_CHECKPOINT = os.path.join(BASE_DIR, "weights", "sam_vit_b_01ec64.pth")
RF_MODEL_PATH = os.path.join(BASE_DIR, "weights", "tomato_rf_model.pkl")

SAM_MODEL_TYPE = "vit_b"

    

def load_all_models():
    """Загружаем все нейронки один раз"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Инициализация моделей. Использую: {device}")
    
    # 1. SAM
    sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
    sam.to(device=device)
    generator = SamAutomaticMaskGenerator(
        model=sam, points_per_side=32, pred_iou_thresh=0.88, 
        stability_score_thresh=0.93, min_mask_region_area=4000
    )
    
    # 2. Random Forest
    rf = joblib.load(RF_MODEL_PATH)
    
    # 3. EasyOCR
    reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
    
    return generator, rf, reader

def surgical_shadow_removal(image, shadow_lift=10, edge_strength=1.5):
    """
    Поднимает только глубокие тени и усиливает границы.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    l_float = l.astype(np.float32) / 255.0
    l_log = np.log1p(l_float * shadow_lift) / np.log1p(shadow_lift)
    l_enhanced = (l_log * 255).astype(np.uint8)

    blur = cv2.GaussianBlur(l_enhanced, (5, 5), 0)
    l_final = cv2.addWeighted(l_enhanced, edge_strength, blur, -(edge_strength - 1.0), 0)

    new_lab = cv2.merge((l_final, a, b))
    result = cv2.cvtColor(new_lab, cv2.COLOR_LAB2BGR)

    return cv2.medianBlur(result, 3)

def load_and_preprocess(image_path):
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise FileNotFoundError(f"Не удалось загрузить: {image_path}")

    image_enhanced = surgical_shadow_removal(image_bgr, shadow_lift=20, edge_strength=3.0)

    lab = cv2.cvtColor(image_enhanced, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.1, tileGridSize=(12, 12))
    l = clahe.apply(l)
    image_final = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

    return {
        'original': cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB),
        'enhanced': cv2.cvtColor(image_final, cv2.COLOR_BGR2RGB),
        'hsv': cv2.cvtColor(image_final, cv2.COLOR_BGR2HSV)
    }

# --- СЕГМЕНТАЦИЯ (Этап 1-2) ---

def is_background_edge(mask, threshold=0.65):
    """
    Твоя логика: проверяем, какую долю границы занимает маска.
    Если маска занимает более 65% любой из сторон — это фон (скатерть/стена).
    Если меньше — это объект, прижатый к краю (например, яйцо).
    """
    h, w = mask.shape[:2]
    top_ratio = np.sum(mask[0, :]) / w
    bottom_ratio = np.sum(mask[-1, :]) / w
    left_ratio = np.sum(mask[:, 0]) / h
    right_ratio = np.sum(mask[:, -1]) / h

    return max(top_ratio, bottom_ratio, left_ratio, right_ratio) > threshold

def get_basic_masks(image_rgb, generator):
    """Первичный проход: только площадь и 'умный' край."""
    h, w = image_rgb.shape[:2]
    raw_masks = generator.generate(image_rgb)

    basic_filtered = []
    for ann in raw_masks:
        mask = ann['segmentation']
        # 1. Фильтр площади
        if ann['area'] > (h * w * 0.3) or ann['area'] < 600:
            continue
        # 2. Твой фильтр границ (теперь не удаляет прижатые объекты)
        if is_background_edge(mask, threshold=0.7):
            continue

        basic_filtered.append(ann)
    return basic_filtered



def calculate_solidity(mask):
    """Коэффициент выпуклости (Solidity)."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return 0
    cnt = contours[0]
    area = cv2.contourArea(cnt)
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    return float(area) / hull_area if hull_area > 0 else 0

def calculate_circularity(mask):
    """Вычисляет круглость объекта (1.0 = идеальный круг)."""
    mask_uint8 = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0

    cnt = contours[0]
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)

    if perimeter == 0:
        return 0

    circularity = (4 * math.pi * area) / (perimeter ** 2)
    return circularity

def get_iou(mask1, mask2):
    """Intersection over Union для проверки дубликатов."""
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union > 0 else 0

def try_merge_masks(masks):
    """Сливает маски, если их объединение увеличивает Solidity."""
    if len(masks) < 2: return masks

    merged_indices = set()
    final_masks = []

    for i in range(len(masks)):
        if i in merged_indices: continue
        best_pair = None

        for j in range(i + 1, len(masks)):
            if j in merged_indices: continue

            m1, m2 = masks[i]['segmentation'], masks[j]['segmentation']

            # Проверка близости (дилатация на 10 пикселей)
            kernel = np.ones((10, 10), np.uint8)
            dilated = cv2.dilate(m1.astype(np.uint8), kernel, iterations=1)
            if np.logical_and(dilated, m2).any():
                s1, s2 = calculate_solidity(m1), calculate_solidity(m2)
                combined = np.logical_or(m1, m2)
                s_total = calculate_solidity(combined)

                # Сливаем, если общее тело стало выпуклым (эллипсом)
                if s_total > max(s1, s2) and s_total > 0.93:
                    best_pair = j
                    break

        if best_pair is not None:
            new_ann = masks[i].copy()
            new_ann['segmentation'] = np.logical_or(masks[i]['segmentation'], masks[best_pair]['segmentation'])
            new_ann['area'] = new_ann['segmentation'].sum()
            final_masks.append(new_ann)
            merged_indices.add(best_pair)
        else:
            final_masks.append(masks[i])
    return final_masks

def adaptive_size_filter(masks, upper_mult=2.0, lower_mult=0.5, circ_threshold=0.82):
    """
    Удаляет аномалии, но щадит крупные идеально круглые объекты.
    """
    if not masks: return []

    areas = [ann['area'] for ann in masks]
    median_area = np.median(areas)

    if median_area < 1000:
        median_area = 2000

    filtered = []
    for ann in masks:
        area = ann['area']
        is_too_big = area > median_area * upper_mult
        is_too_small = area < median_area * lower_mult

        # Если объект слишком большой, проверяем его на круглость
        if is_too_big:
            circ = calculate_circularity(ann['segmentation'])
            # Если он круглый — прощаем ему размер
            if circ > circ_threshold or (area >= 2150 and area<=2400):
                filtered.append(ann)
                continue
            else:
                # Если большой и не круглый (например, кусок фона) — удаляем
                continue

        # Если не слишком большой, проверяем только нижний порог
        if not is_too_small:
            filtered.append(ann)

    return filtered

def is_border_artifact(mask, border_thickness=5, max_touch_ratio=0.25):
    """
    Удаляет куски фона по краям.
    Если объект прилип к границе кадра более чем на 25% своего периметра - в топку.
    """
    H, W = mask.shape
    mask_uint8 = mask.astype(np.uint8)

    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return False

    # Берем самый большой контур на случай микро-шума в маске
    cnt = max(contours, key=cv2.contourArea)

    # 1. Рисуем ТОЛЬКО КОНТУР объекта (толщина строго 1 пиксель)
    perimeter_mask = np.zeros_like(mask_uint8)
    cv2.drawContours(perimeter_mask, [cnt], -1, 1, 1)

    # Считаем длину периметра в пикселях (теперь это точное число)
    total_perimeter_pixels = perimeter_mask.sum()
    if total_perimeter_pixels == 0: return False

    # 2. Создаем рамку по краям изображения
    border_mask = np.zeros_like(mask_uint8)
    cv2.rectangle(border_mask, (0, 0), (W - 1, H - 1), 1, border_thickness)

    # 3. Пересекаем КОНТУР объекта с РАМКОЙ
    touch_pixels = np.logical_and(perimeter_mask, border_mask).sum()

    # 4. Теперь мы честно делим пиксели контура на пиксели контура
    ratio = touch_pixels / total_perimeter_pixels

    return ratio > max_touch_ratio


def is_too_elongated(mask, max_ratio=3.5):
    """Проверяет, не является ли объект слишком вытянутой 'полоской'."""
    mask_uint8 = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False

    cnt = max(contours, key=cv2.contourArea)
    # Вписываем ориентированный прямоугольник (может быть под углом)
    rect = cv2.minAreaRect(cnt)
    width, height = rect[1]

    if width == 0 or height == 0:
        return True # Совсем тонкие объекты удаляем

    ratio = max(width, height) / min(width, height)
    return ratio > max_ratio


def smooth_mask_to_convex(mask):
    """Превращает рваную маску в идеально выпуклую форму (Convex Hull)."""
    # 1. Приводим маску к uint8, так как OpenCV не ест bool
    mask_uint8 = mask.astype(np.uint8)

    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask

    # 2. Берем самый большой контур
    cnt = max(contours, key=cv2.contourArea)

    # 3. Строим выпуклую оболочку
    hull = cv2.convexHull(cnt)

    # 4. Создаем пустую маску именно в формате uint8 (нули)
    new_mask = np.zeros(mask.shape, dtype=np.uint8)

    # 5. Рисуем. Теперь типы данных совпадают!
    cv2.drawContours(new_mask, [hull], -1, 1, thickness=cv2.FILLED)

    # 6. Возвращаем обратно в bool, если тебе так удобнее для дальнейших расчетов
    return new_mask.astype(bool)

def is_shadow_by_background_match(image_orig_rgb, mask, brightness_ratio_max=0.88, color_diff_max=4.5, circ_threshold=0.8):
    """
    Сравнение объекта с его локальным фоном + иммунитет для еды.
    """
    # --- 1. ПРОВЕРКА ИММУНИТЕТА (ОТХОДНЫЕ ПУТИ) ---
    area = np.sum(mask)

    # Путь А: Иммунитет по площади (Золотое яйцо)
    if 2150 <= area <= 2500:
        return False # Это точно не тень

    # Путь Б: Иммунитет по форме (Круглые помидоры)
    # Используем твою существующую функцию calculate_circularity
    circ = calculate_circularity(mask)
    if circ > circ_threshold:
        return False # Слишком круглое для тени

    # --- 2. ЕСЛИ ИММУНИТЕТА НЕТ, АНАЛИЗИРУЕМ ЦВЕТ ---
    lab = cv2.cvtColor(image_orig_rgb, cv2.COLOR_RGB2LAB)

    # Получаем кольцо фона
    kernel = np.ones((9, 9), np.uint8)
    dilated = cv2.dilate(mask.astype(np.uint8), kernel, iterations=2)
    bg_mask = (dilated == 1) & (mask == 0)

    if not np.any(bg_mask) or not np.any(mask):
        return False

    mean_obj = np.mean(lab[mask], axis=0) # [L, A, B]
    mean_bg = np.mean(bg_mask_pixels := lab[bg_mask], axis=0)

    # Расстояние цвета (отличие оттенка от фона)
    color_dist = math.sqrt((mean_obj[1] - mean_bg[1])**2 + (mean_obj[2] - mean_bg[2])**2)

    # Коэффициент яркости (насколько объект темнее фона)
    brightness_ratio = mean_obj[0] / mean_bg[0] if mean_bg[0] > 0 else 1.0

    # ВЕРДИКТ: Если объект похож по цвету на фон и он темный
    if color_dist < color_diff_max and brightness_ratio < brightness_ratio_max:
        return True

    return False


def refine_masks(image_rgb, basic_masks):
    step1 = adaptive_size_filter(basic_masks, upper_mult=2.0, lower_mult=0.5)

    step2 = try_merge_masks(step1)

    for ann in step2:
        ann['segmentation'] = smooth_mask_to_convex(ann['segmentation'])
        ann['area'] = ann['segmentation'].sum()

    step3 = adaptive_size_filter(step2, upper_mult=1.4, lower_mult=0.2)

    sorted_masks = sorted(step3, key=lambda x: x['area'], reverse=True)
    refined = []

    for ann in sorted_masks:
        mask = ann['segmentation']
        area_current = ann['area']
        if calculate_solidity(mask) < 0.85:
            continue

        if is_too_elongated(mask, max_ratio=3.0):
            continue

        if is_border_artifact(mask, border_thickness=5, max_touch_ratio=0.5):
            continue


        if is_shadow_by_background_match(image_rgb, mask, brightness_ratio_max=0.95, color_diff_max=4.0):
           continue

        is_nested_or_duplicate = False
        for saved in refined:
            # Считаем пересечение текущей (меньшей) маски с уже сохраненной (большей)
            intersection = np.logical_and(mask, saved['segmentation']).sum()
            
            # 1. Коэффициент вхождения (Containment)
            # Сколько процентов текущей маски покрыто сохраненной большой маской
            containment = intersection / (area_current + 1e-6)
            
            # 2. Твой старый IoU (для просто пересекающихся объектов)
            union = area_current + saved['area'] - intersection
            iou = intersection / (union + 1e-6)

            # Если маска на 70% внутри другой или имеет высокий IoU — это матрешка
            if containment > 0.7 or iou > 0.6:
                is_nested_or_duplicate = True
                break

        if not is_nested_or_duplicate:
            refined.append(ann)

    return refined

# Классификация 

def predict_and_process(image_rgb, masks, model, feat_columns):
    results = []
    
    # Подготовка слоев
    image_hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    image_lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    
    # Фон для относительных признаков
    all_objs_mask = np.zeros(image_rgb.shape[:2], dtype=np.uint8)
    for ann in masks:
        all_objs_mask = cv2.bitwise_or(all_objs_mask, ann['segmentation'].astype(np.uint8))
    bg_mask = cv2.bitwise_not(all_objs_mask)
    
    bg_rg_ratio, bg_v_med, bg_s_med = 1.0, 120.0, 40.0
    if np.sum(bg_mask) > 100:
        bg_pixels = image_rgb[bg_mask > 0]
        bg_rg_ratio = np.median(bg_pixels[:, 0]) / (np.median(bg_pixels[:, 1]) + 1e-6)
        bg_v_med = np.median(image_hsv[bg_mask > 0][:, 2])
        bg_s_med = np.median(image_hsv[bg_mask > 0][:, 1])

    # Текстура
    sobelx = cv2.Sobel(image_gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(image_gray, cv2.CV_64F, 0, 1, ksize=3)
    mag_img = np.sqrt(sobelx**2 + sobely**2)

    for ann in masks:
        mask = ann['segmentation'].astype(np.uint8)
        if np.sum(mask) < 30: continue
        
        # СБОР ТОЧНО ТАКИХ ЖЕ ПРИЗНАКОВ, КАК В ТАБЛИЦЕ
        px_rgb = image_rgb[mask > 0]; px_hsv = image_hsv[mask > 0]
        px_lab = image_lab[mask > 0]; px_gray = image_gray[mask > 0]
        
        r_med, g_med, b_med = np.median(px_rgb, axis=0)
        h_med, s_med, v_med = np.median(px_hsv, axis=0)
        l_med, a_med, bl_med = np.median(px_lab, axis=0)
        
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnt = cnts[0]; area = cv2.contourArea(cnt); peri = cv2.arcLength(cnt, True)
        
        # Формируем словарь признаков (строго по списку!)
        feat_dict = {
            'r_med': r_med, 'g_med': g_med, 'b_med': b_med,
            'h_med': h_med, 's_med': s_med, 'v_med': v_med,
            'l_med': l_med, 'a_med': a_med, 'bl_med': bl_med,
            'rg_ratio': r_med / (g_med + 1e-6),
            'rg_diff_bg': (r_med / (g_med + 1e-6)) - bg_rg_ratio,
            'v_ratio_bg': v_med / (bg_v_med + 1e-6),
            's_diff_bg': s_med - bg_s_med,
            'std_val': np.std(px_gray),
            'edge_energy': np.mean(mag_img[mask > 0]),
            'area': area,
            'circularity': (4 * np.pi * area) / (peri**2 + 1e-6),
            'solidity': area / (cv2.contourArea(cv2.convexHull(cnt)) + 1e-6)
        }
        
        # Превращаем в DataFrame для модели
        row_df = pd.DataFrame([feat_dict])[feat_columns]
        
        # Предсказание вероятности
        probs = model.predict_proba(row_df)[0]
        max_idx = np.argmax(probs)
        label = model.classes_[max_idx]
        confidence = probs[max_idx]

        # --- СТРАХОВКА ОТ ОШИБОК ---
        # Если модель не уверена (меньше 50%) — скорее всего это тень или мусор
        if confidence < 0.50:
            label = "Shadow"

        results.append({
            'mask': mask, 'label': label, 'cnt': cnt, 'conf': confidence
        })
        
    return results

def draw_segmentation_only(image_rgb, objects):
    """Окно 1: Только маски (результат работы SAM)"""
    overlay = image_rgb.copy()
    # Рисуем все найденные объекты одним нейтральным цветом (например, белым), 
    # чтобы показать именно работу сегментатора
    for obj in objects:
        mask = obj['mask']
        overlay[mask > 0] = (255, 255, 255) 
    
    return cv2.addWeighted(image_rgb, 0.6, overlay, 0.4, 0)

def draw_classification_only(image_rgb, objects):
    """Окно 2: Результат классификации (цвета по классам)"""
    output = image_rgb.copy()
    
    # Твоя палитра
    colors = {
        "Red Tomato": (255, 0, 0),      # Красный
        "Orange Tomato": (255, 165, 0), # Оранжевый
        "Quail Egg": (64, 224, 208),    # Бирюзовый
        "Shadow": (128, 128, 128)       # Серый
    }

    for obj in objects:
        cnt = obj['cnt']
        label = obj['label']
        color = colors.get(label, (255, 255, 255))
        
        # Только обводка (без текста)
        cv2.drawContours(output, [cnt], -1, color, 3)
                    
    return output

def reconstruct_letter_final_boss(image_shape, objects, reader):
    # 1. Сбор маски всех ингредиентов
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    food_classes = ["Red Tomato", "Orange Tomato", "Quail Egg"]
    
    found_food = False
    for obj in objects:
        if obj['label'] in food_classes:
            mask = cv2.bitwise_or(mask, (obj['mask'] * 255).astype(np.uint8))
            found_food = True
    
    if not found_food:
        return np.ones(image_shape[:2], dtype=np.uint8) * 255, "?"

    # 2. Склеивание в единый массив (жирно, чтобы сраслось)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (80, 80))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
    
    # 3. Скелетизация (поиск "хребта" буквы)
    # Сгладим перед скелетом, чтобы было меньше "веток"
    closed = cv2.GaussianBlur(closed, (15, 15), 0)
    _, binary = cv2.threshold(closed, 127, 255, cv2.THRESH_BINARY)
    skel = skeletonize(binary > 0)
    skel_img = (skel * 255).astype(np.uint8)

    # 4. Геометрическое выпрямление и тонкая отрисовка
    final_canvas = np.ones(image_shape[:2], dtype=np.uint8) * 255
    cnts, _ = cv2.findContours(skel_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    target_thickness = 25 # Тонкая линия для OCR

    for cnt in cnts:
        # Аппроксимация: epsilon 0.02 - 0.04 для прямых линий
        epsilon = 0.025 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, closed=False)
        
        if len(approx) < 2: continue

        # Рисуем сегменты как прямоугольные балки
        for i in range(len(approx) - 1):
            p1 = tuple(approx[i][0])
            p2 = tuple(approx[i+1][0])
            
            # Маска для сегмента, чтобы сделать его прямоугольным
            line_mask = np.zeros(image_shape[:2], dtype=np.uint8)
            cv2.line(line_mask, p1, p2, 255, target_thickness)
            
            l_cnts, _ = cv2.findContours(line_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if l_cnts:
                rect = cv2.minAreaRect(l_cnts[0])
                box = cv2.boxPoints(rect).astype(np.int64) # Исправленный int0
                cv2.drawContours(final_canvas, [box], 0, 0, -1)

    # 5. Подготовка к распознаванию (Паддинг + Размытие)
    # Добавляем 200 пикселей белого поля
    padded = cv2.copyMakeBorder(final_canvas, 200, 200, 200, 200, cv2.BORDER_CONSTANT, value=255)
    
    # Легкое размытие, чтобы детектор текста CRAFT лучше видел линии
    for_ocr = cv2.GaussianBlur(padded, (3, 3), 0)
    height = 600
    width = int(for_ocr.shape[1] * (height / for_ocr.shape[0]))
    for_ocr_resized = cv2.resize(for_ocr, (width, height), interpolation=cv2.INTER_AREA)
    # 6. Распознавание
    result = reader.readtext(
        for_ocr_resized, 
        detail=0, 
        slope_ths=0.5,
        allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ',
        paragraph=True # Помогает, если буква одна большая
    )
    char = result[0] if result else "?"
    
    # Возвращаем final_canvas (без паддингов для визуалки) и символ
    return final_canvas, char