# -*- coding: utf-8 -*-
"""
tree_detector.py
================
Ядро детекции деревьев. БЕЗ кликов, БЕЗ захвата экрана.
Тут только логика: дать картинку -> вернуть список деревьев.

Идея детекции (то что обсудили):
  Дерево = зелёная крона СВЕРХУ + коричневый ствол СНИЗУ.
  Ствол стоит на траве -> граница "коричневый над зелёным".
  Это отсекает: траву (нет ствола), дорогу (нет зелёной кроны сверху),
  брёвна на земле (лежат, нет вертикали крона->ствол).

Все числа (HSV-диапазоны, пороги) вынесены в CONFIG наверху —
крутим их на реальном debug-фото, код не трогаем.
"""

import cv2
import numpy as np


# ============================================================
# CONFIG — крутим тут
# ============================================================

# --- Диапазоны цвета в HSV (OpenCV: H 0-179, S 0-255, V 0-255) ---
# ВАЖНО: это маска КРОНЫ, не травы.
# Замеры: газон V~169 (яркий), крона дерева V~117-153 (ТЁМНЫЙ зелёный).
# Берём только тёмный зелёный (V<=150) -> газон выпадает, крона остаётся.
# Над лужайкой-пятном сверху яркий газон -> не пройдёт. Над деревом крона -> пройдёт.
GREEN_LOWER = (32, 60, 40)
GREEN_UPPER = (90, 255, 150)

# Коричневый ствол. Подобрано по реальным замерам:
#   ствол  H27-39 S180-202 V90-120  -> ловим
#   дом    H17-18 S77-92            -> выкид (H ниже, S ниже)
#   земля/дорога                    -> выкид (низкая S или зелёный)
BROWN_LOWER = (22, 150, 60)
BROWN_UPPER = (42, 255, 150)

# --- Фильтры контура ствола ---
MIN_TRUNK_AREA = 8       # меньше площади -> шум, выкид (стволы тонкие)
MAX_TRUNK_AREA = 4000    # больше -> дом/дорога/большое пятно, выкид
MAX_TRUNK_WIDTH = 60     # ствол узкий; шире -> не ствол

# --- Проверка "зелёная крона над стволом" ---
# Над коробкой ствола смотрим полосу высотой CROWN_LOOK_UP (в % от высоты ствола)
# и шире ствола в стороны. Если зелёных пикселей в ней >= CROWN_MIN_RATIO -> дерево.
CROWN_LOOK_UP_FACTOR = 2.5   # высота зоны кроны = высота ствола * это
CROWN_SIDE_PAD = 6           # расширить зону кроны в стороны, px
CROWN_MIN_RATIO = 0.30       # доля ТЁМНОЙ кроны в зоне чтоб засчитать
CROWN_MIN_AREA = 60          # абсолют: тёмной кроны над стволом не меньше, px

# --- Куда кликать: 'crown' (центр кроны) или 'base' (основание ствола) ---
# Игра принимает клик и по кроне -> 'crown' проще/надёжнее (не зависит от
# поиска тонкого стыка ствол-трава).
CLICK_TARGET = 'crown'

# Для 'base': от низа ствола сканим ВНИЗ до первой травы = земля = основание.
GRASS_LOWER = (32, 40, 60)    # широкий зелёный (трава+крона) для скана вниз
GRASS_UPPER = (90, 255, 255)
TRUNK_DROP_MAX = 30           # макс. сколько px сканить вниз в поисках травы
TRUNK_BASE_NUDGE = 0          # сдвиг итоговой точки вниз (в траву), px

# --- Морфология (чистка масок) ---
MORPH_KERNEL = 3

# --- Зоны UI которые игнорим (доля от ширины/высоты экрана) ---
# (x1, y1, x2, y2) в долях 0..1. Всё что попало сюда — не дерево.
IGNORE_ZONES = [
    (0.0, 0.88, 1.0, 1.0),    # нижняя полоса (инвентарь/хотбар)
    (0.0, 0.70, 0.26, 1.0),   # чат слева снизу
    (0.70, 0.0, 1.0, 0.22),   # миникарта справа сверху
]


# ============================================================
# Детекция
# ============================================================

def _build_masks(bgr):
    """BGR -> HSV -> маски: crown(тёмный зелёный), brown(ствол), grass(широкий зелёный)."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    green = cv2.inRange(hsv, np.array(GREEN_LOWER), np.array(GREEN_UPPER))
    brown = cv2.inRange(hsv, np.array(BROWN_LOWER), np.array(BROWN_UPPER))
    grass = cv2.inRange(hsv, np.array(GRASS_LOWER), np.array(GRASS_UPPER))

    k = np.ones((MORPH_KERNEL, MORPH_KERNEL), np.uint8)
    # Зелёный — большие площади, open убирает мелкий шум.
    green = cv2.morphologyEx(green, cv2.MORPH_OPEN, k)
    # Ствол ТОНКИЙ (1-2px). OPEN его сотрёт! Только close+dilate —
    # склеиваем разрозненные пиксели ствола в одно пятно.
    brown = cv2.morphologyEx(brown, cv2.MORPH_CLOSE, k)
    brown = cv2.dilate(brown, k, iterations=1)

    return green, brown, grass


def _trunk_base(grass, cx, box_bottom, W, H):
    """
    Основание ствола = первый травяной пиксель ВНИЗ от низа коричневого пятна.
    Крона выше, трава ниже -> первый зелёный снизу = земля.
    Возвращает y основания (или box_bottom если травы не нашли).
    """
    col = min(max(cx, 0), W - 1)
    for dy in range(0, TRUNK_DROP_MAX):
        yy = box_bottom + dy
        if yy >= H:
            break
        if grass[yy, col]:
            return yy + TRUNK_BASE_NUDGE
    return box_bottom


def _in_ignore_zone(cx, cy, W, H):
    """Точка попала в UI-зону?"""
    for (x1, y1, x2, y2) in IGNORE_ZONES:
        if x1 * W <= cx <= x2 * W and y1 * H <= cy <= y2 * H:
            return True
    return False


# Подсветка кликнутого дерева = ЯРКИЙ салатовый глоу (V>=195).
# Замеры: глоу V206-253 neon~0.6-0.9; трава/обычные деревья neon=0.00.
# Держится весь руб -> идеальный сигнал "дерево ещё рубится".
HIGHLIGHT_LOWER = (35, 40, 195)
HIGHLIGHT_UPPER = (90, 255, 255)


def highlight_count(bgr, cx, cy, r):
    """
    Кол-во пикселей ЯРКОЙ подсветки в зоне вокруг (cx,cy) радиусом r.
    Абсолют надёжнее доли: большой глоу рубки -> сотни px; трава/перс -> ~0.
    Камера едет за персом -> зону берём вокруг ПЕРСА (центр), не у дерева.
    """
    H, W = bgr.shape[:2]
    x1, x2 = max(0, cx - r), min(W, cx + r)
    y1, y2 = max(0, cy - r), min(H, cy + r)
    if x2 <= x1 or y2 <= y1:
        return 0
    hsv = cv2.cvtColor(bgr[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
    neon = cv2.inRange(hsv, np.array(HIGHLIGHT_LOWER), np.array(HIGHLIGHT_UPPER))
    return int(np.count_nonzero(neon))


def detect_trees(bgr):
    """
    Главная функция. Вход: картинка BGR (как из OpenCV).
    Выход: список деревьев. Каждое:
        {
          'box':   (x, y, w, h),     # коробка ствола
          'click': (cx, cy),         # куда кликать (низ ствола)
          'crown_ratio': float,      # доля зелёного над стволом (для отладки)
        }
    Плюс возвращаем (rejected, green_mask, brown_mask) для debug-картинки.
    """
    H, W = bgr.shape[:2]
    green, brown, grass = _build_masks(bgr)

    contours, _ = cv2.findContours(brown, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    trees = []
    rejected = []   # (box, причина) — рисуем красным в debug

    for c in contours:
        area = cv2.contourArea(c)
        x, y, w, h = cv2.boundingRect(c)

        # --- фильтр площади ствола ---
        if area < MIN_TRUNK_AREA:
            continue  # мелкий шум, даже не рисуем
        if area > MAX_TRUNK_AREA or w > MAX_TRUNK_WIDTH:
            rejected.append(((x, y, w, h), "size"))
            continue

        # точка клика по X = центр ствола; по Y = ОСНОВАНИЕ (стык с травой).
        # Сканим вниз от низа коробки до первой травы — точно в землю, не в крону.
        cx = x + w // 2
        cy = _trunk_base(grass, cx, y + h, W, H)

        # --- UI-зоны ---
        if _in_ignore_zone(cx, cy, W, H):
            rejected.append(((x, y, w, h), "ui"))
            continue

        # --- проверка зелёной кроны НАД стволом ---
        look_up = int(h * CROWN_LOOK_UP_FACTOR)
        ry1 = max(0, y - look_up)
        ry2 = y
        rx1 = max(0, x - CROWN_SIDE_PAD)
        rx2 = min(W, x + w + CROWN_SIDE_PAD)

        crown_ratio = 0.0
        crown_area = 0
        if ry2 > ry1 and rx2 > rx1:
            roi = green[ry1:ry2, rx1:rx2]
            if roi.size > 0:
                crown_area = int(np.count_nonzero(roi))
                crown_ratio = crown_area / roi.size

        # нужна И доля И абсолютная площадь тёмной кроны
        if crown_ratio < CROWN_MIN_RATIO or crown_area < CROWN_MIN_AREA:
            rejected.append(((x, y, w, h), "no_crown"))
            continue

        # --- итоговая точка клика ---
        if CLICK_TARGET == 'crown':
            # центр масс зелёной кроны над стволом
            ys, xs = np.nonzero(roi)
            if len(xs) > 0:
                cx = rx1 + int(xs.mean())
                cy = ry1 + int(ys.mean())
            # иначе остаётся точка ствола (fallback)

        trees.append({
            'box': (x, y, w, h),
            'click': (cx, cy),
            'crown_ratio': crown_ratio,
            'crown_area': crown_area,      # площадь тёмной кроны над стволом
            'trunk_area': int(area),       # площадь пятна ствола
        })

    return trees, rejected, green, brown
