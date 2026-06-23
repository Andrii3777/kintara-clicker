# -*- coding: utf-8 -*-
"""
stone_detector.py
=================
Ядро детекции КАМНЕЙ и МЕТАЛЛА. БЕЗ кликов, БЕЗ захвата экрана.
Дать картинку BGR -> вернуть список камней / металла.

Идея детекции (по замерам реальных пикселей примеров в images-examples/):
  Камень = СЕРЫЙ/бежевый воксель-блоб на траве. Главный признак — НИЗКАЯ
  насыщенность (S~34-51) при тёплом H~20-26 и средней V~60-110. Трава наоборот
  яркая и насыщенная (S>125), ствол дерева того же H но S>150. Значит камень
  ловится "тёплый серый = низкая S".

  Металл = похожий серый блоб, но лежит на СНЕГУ (белый фон, S≈0, V>180).
  Проверка окружения: снег вместо травы.

  Проблема: в городе серого МНОГО (крыши, брусчатка, грунт, заборы). Их
  отсекаем тем что камень = КОМПАКТНЫЙ блоб, ОКРУЖЁННЫЙ ТРАВОЙ. Брусчатка/крыша
  большие и окружены не травой (аналог проверки "крона над стволом" у дерева).

Все числа в CONFIG сверху — крутим на debug_detect.py, код не трогаем.
"""

import cv2
import numpy as np


# ============================================================
# CONFIG — крутим тут
# ============================================================

# --- Цвет КАМНЯ + УГЛЯ в HSV (OpenCV: H 0-179, S 0-255, V 0-255) ---
# Ловим ОБА ресурса (добываются одинаково). Замеры:
#   камень H20-26 S34-51 V72-110 (медиана ~88)
#   уголь  H12-28 S35-66 V42-101 (медиана ~60, ТЕМНЕЕ камня)
# Общая сигнатура: тёплый H, НИЗКАЯ S (трава S>125 выпадает), V 40-150.
# Нижний V=40 включает тёмный уголь (раньше было 72 — уголь отсекался).
STONE_LOWER = (5, 18, 40)
STONE_UPPER = (35, 70, 150)

# --- Трава (для проверки "камень окружён травой") ---
# Яркий насыщенный зелёный. Замеры: H49-57 S125-230 V100-186.
GRASS_LOWER = (32, 60, 60)
GRASS_UPPER = (90, 255, 255)

# --- Морфология ---
MORPH_KERNEL = 3
STONE_CLOSE_ITERS = 1     # склеить блок камня в пятно. 1 а не 2: 2 слипал
                          # соседние камни через траву в рваный блоб -> shape-отказ.
STONE_OPEN_ITERS = 1      # убрать мелкий серый шум (травинки/тропинки в 1px)

# --- Фильтры контура камня ---
MIN_STONE_AREA = 150      # меньше -> шум/крошка/стыки тайлов/края брусчатки, выкид
MAX_STONE_AREA = 7500     # больше -> дом/строение/брусчатка, выкид. Замер:
                          # дома ~9400-9700, песчаный холм/формация ~6300 ->
                          # порог между ними. Холм (добываемый) проходит.
MIN_EXTENT = 0.35         # площадь/площадь_бокса: камень плотный; тонкие/рваные выкид
MAX_ASPECT = 3.0          # w/h или h/w больше -> полоса (тропинка/край крыши), выкид

# --- Проверка "камень рядом с травой" (МЯГКАЯ) ---
# Камни живут В ПРИРОДЕ -> вокруг трава. В природе конфьюзеров нет (трава/дерево
# отсеяны цветом), поэтому гейт МЯГКИЙ: бьём только то у чего травы вообще нет
# (крупная брусчатка/крыша в городе). Песчаный холм/кластер у деревьев — трава
# хотя бы по краю -> проходят.
RING_PAD = 8              # ширина кольца вокруг бокса, px
GRASS_RING_MIN = 0.06    # доля травы в кольце чтоб засчитать камень

# --- Проверка ТЕКСТУРЫ (отсев затенённых граней зелёных блоков / теней) ---
# Затенённая боковая грань воксель-блока травы/кроны по ЦВЕТУ = камень (H/S/V
# совпадают, замерено), но ОДНОРОДНАЯ. Настоящий камень/уголь текстурный
# (светлые+тёмные куски, у угля чёрные крапины). Разброс внутри блоба:
#   ложная грань/тень: Vstd низкий И Sstd низкий (всё однородно)
#   камень:            Vstd высокий (светотень кусков)
#   тёмный уголь:      Vstd низкий, НО Sstd высокий (чёрные+серые куски)
# -> выкид только если ОБА низкие. Замер: ложь (8/4, 5/1), уголь-1 (7/8).
STONE_VSTD_MIN = 9.5     # порог разброса яркости
STONE_SSTD_MIN = 5.5     # порог разброса насыщенности

# --- Цвет МЕТАЛЛА в HSV ---
# Металл = ХОЛОДНО-СЕРЫЙ блоб на снегу. Замеры пикселей рудных блобов:
#   H=105-120, S=15-50, V=75-115  (холодный/синеватый серый, темнее снега)
# Снег: H=100-115, S=0-35, V=175-215 (ярче, ниже S)
# Ключ: V<145 отделяет металл от снега (V>165).
METAL_LOWER = (95, 12, 65)
METAL_UPPER = (128, 65, 145)

# --- Снег (для проверки "металл окружён снегом") ---
# Снег: холодный, ахроматический, очень яркий. H=95-120, S=0-40, V>165.
SNOW_LOWER = (95,  0, 162)
SNOW_UPPER = (125, 42, 255)

# --- Фильтры контура МЕТАЛЛА (отдельные от камня — блобы металла мельче) ---
MIN_METAL_AREA = 60    # металл мелкий (60-300 px в типичном разрешении)
MAX_METAL_AREA = 600   # больше — UI/строение (счётчик сервера -> выкид)

# --- Проверка "металл рядом со снегом" (аналог GRASS_RING_MIN) ---
# Металл часто в кластерах -> соседний блоб не даёт снега в кольце.
# Низкий порог 0.03 — хватит угла кластера.
SNOW_RING_MIN = 0.05   # доля снега в кольце вокруг бокса металла

# --- Проверка "рядом НЕТ здания" ---
# Бочки/ящики/основание здания: в кольце тёмные СЕРЫЕ пиксели (V<75, S<40).
# Стволы деревьев тоже тёмные (V<80), но КОРИЧНЕВЫЕ (S≈100-200) -> не попадают.
# Проверяем ОБА условия: V < BUILDING_DARK_V AND S < BUILDING_DARK_S
# Замеры у здания: тёмно-серые стены S=10-30 -> дают dark_ring>0.09.
# Реальный металл у деревьев: стволы S>80 -> не считаются -> dark_ring≈0.
BUILDING_DARK_V = 75       # V ниже этого = тёмный пиксель
BUILDING_DARK_S = 40       # AND S ниже этого = именно серый/каменный, не коричневый ствол
DARK_RING_MAX = 0.09       # порог доли здание-пикселей в кольце
DARK_SNOW_MIN = 0.50       # если snow >= этого — достаточно открытая поляна, не здание

# --- Зоны UI которые игнорим (доли 0..1: x1,y1,x2,y2) ---
IGNORE_ZONES = [
    (0.0, 0.74, 1.0, 1.0),    # хотбар снизу (на этом мониторе 2560x1600 hotbar при y≈75%)
    (0.78, 0.0, 1.0, 0.20),   # миникарта справа сверху
    (0.0, 0.0, 0.12, 0.06),   # ярлык сервера слева сверху
    (0.88, 0.10, 1.0, 0.55),  # колонка иконок справа (расширено с 0.92)
]


# ============================================================
# Детекция
# ============================================================

def _build_masks(bgr):
    """BGR -> HSV -> маски: stone (серый блоб), grass (для проверки окружения)."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    stone = cv2.inRange(hsv, np.array(STONE_LOWER), np.array(STONE_UPPER))
    grass = cv2.inRange(hsv, np.array(GRASS_LOWER), np.array(GRASS_UPPER))

    k = np.ones((MORPH_KERNEL, MORPH_KERNEL), np.uint8)
    stone = cv2.morphologyEx(stone, cv2.MORPH_CLOSE, k, iterations=STONE_CLOSE_ITERS)
    stone = cv2.morphologyEx(stone, cv2.MORPH_OPEN, k, iterations=STONE_OPEN_ITERS)
    return stone, grass


def _in_ignore_zone(cx, cy, W, H):
    for (x1, y1, x2, y2) in IGNORE_ZONES:
        if x1 * W <= cx <= x2 * W and y1 * H <= cy <= y2 * H:
            return True
    return False


def _grass_ring_ratio(grass, box, W, H):
    """Доля травы в кольце вокруг бокса камня (камень в траве -> высокая)."""
    x, y, w, h = box
    rx1, ry1 = max(0, x - RING_PAD), max(0, y - RING_PAD)
    rx2, ry2 = min(W, x + w + RING_PAD), min(H, y + h + RING_PAD)
    outer = grass[ry1:ry2, rx1:rx2]
    if outer.size == 0:
        return 0.0
    # вычитаем внутренний бокс — считаем только кольцо
    inner_area = w * h
    ring_area = outer.size - inner_area
    if ring_area <= 0:
        return float(np.count_nonzero(outer)) / outer.size
    ix1, iy1 = x - rx1, y - ry1
    ring_grass = int(np.count_nonzero(outer)) - \
        int(np.count_nonzero(outer[iy1:iy1 + h, ix1:ix1 + w]))
    return max(0.0, ring_grass) / ring_area


def detect_stones(bgr):
    """
    Вход: картинка BGR. Выход: (stones, rejected, stone_mask, grass_mask).
    Каждый камень: {'box':(x,y,w,h), 'click':(cx,cy), 'area':int,
                    'extent':float, 'grass_ring':float}.
    """
    H, W = bgr.shape[:2]
    stone, grass = _build_masks(bgr)
    hsv_full = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)        # для текстуры
    vchan = hsv_full[:, :, 2]
    schan = hsv_full[:, :, 1]

    contours, _ = cv2.findContours(stone, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    stones = []
    rejected = []

    for c in contours:
        area = cv2.contourArea(c)
        x, y, w, h = cv2.boundingRect(c)

        if area < MIN_STONE_AREA:
            continue
        if area > MAX_STONE_AREA:
            rejected.append(((x, y, w, h), "size"))
            continue

        extent = area / float(w * h) if w * h > 0 else 0.0
        aspect = max(w / float(h), h / float(w)) if w > 0 and h > 0 else 99
        if extent < MIN_EXTENT or aspect > MAX_ASPECT:
            rejected.append(((x, y, w, h), "shape"))
            continue

        cx, cy = x + w // 2, y + h // 2
        if _in_ignore_zone(cx, cy, W, H):
            rejected.append(((x, y, w, h), "ui"))
            continue

        gring = _grass_ring_ratio(grass, (x, y, w, h), W, H)
        if gring < GRASS_RING_MIN:
            rejected.append(((x, y, w, h), "no_grass"))
            continue

        # текстура: ОБА разброса низкие = затенённая грань блока/тень -> выкид.
        # (камень: Vstd высок; тёмный уголь: Vstd низок но Sstd высок)
        roi_mask = stone[y:y + h, x:x + w]
        roi_v = vchan[y:y + h, x:x + w][roi_mask > 0]
        roi_s = schan[y:y + h, x:x + w][roi_mask > 0]
        vstd = float(np.std(roi_v)) if roi_v.size >= 5 else 0.0
        sstd = float(np.std(roi_s)) if roi_s.size >= 5 else 0.0
        if vstd < STONE_VSTD_MIN and sstd < STONE_SSTD_MIN:
            rejected.append(((x, y, w, h), "flat"))
            continue

        stones.append({
            'box': (x, y, w, h),
            'click': (cx, cy),
            'area': int(area),
            'extent': round(extent, 2),
            'grass_ring': round(gring, 2),
            'vstd': round(vstd, 1),
            'sstd': round(sstd, 1),
        })

    return stones, rejected, stone, grass


# ============================================================
# Детекция МЕТАЛЛА (серый блоб на снегу)
# ============================================================

def _dark_ring_ratio(vchan, schan, box, W, H):
    """Доля пикселей ЗДАНИЯ в кольце: тёмные (V<BUILDING_DARK_V) И серые (S<BUILDING_DARK_S).
    Стволы деревьев тёмные но коричневые (S>80) -> не считаются -> метал у деревьев не режется.
    """
    x, y, w, h = box
    rx1, ry1 = max(0, x - RING_PAD), max(0, y - RING_PAD)
    rx2, ry2 = min(W, x + w + RING_PAD), min(H, y + h + RING_PAD)
    outer_v = vchan[ry1:ry2, rx1:rx2]
    outer_s = schan[ry1:ry2, rx1:rx2]
    if outer_v.size == 0:
        return 0.0
    inner_area = w * h
    ring_area = outer_v.size - inner_area
    if ring_area <= 0:
        return 0.0
    ix1, iy1 = x - rx1, y - ry1
    building = (outer_v < BUILDING_DARK_V) & (outer_s < BUILDING_DARK_S)
    building_outer = int(np.sum(building))
    building_inner = int(np.sum(building[iy1:iy1 + h, ix1:ix1 + w]))
    return max(0.0, building_outer - building_inner) / ring_area


def _snow_ring_ratio(snow, box, W, H):
    """Доля снега в кольце вокруг бокса металла (металл на снегу -> высокая)."""
    x, y, w, h = box
    rx1, ry1 = max(0, x - RING_PAD), max(0, y - RING_PAD)
    rx2, ry2 = min(W, x + w + RING_PAD), min(H, y + h + RING_PAD)
    outer = snow[ry1:ry2, rx1:rx2]
    if outer.size == 0:
        return 0.0
    inner_area = w * h
    ring_area = outer.size - inner_area
    if ring_area <= 0:
        return float(np.count_nonzero(outer)) / outer.size
    ix1, iy1 = x - rx1, y - ry1
    ring_snow = int(np.count_nonzero(outer)) - \
        int(np.count_nonzero(outer[iy1:iy1 + h, ix1:ix1 + w]))
    return max(0.0, ring_snow) / ring_area


def detect_metals(bgr):
    """
    Вход: картинка BGR. Выход: (metals, rejected, metal_mask, snow_mask).
    Каждый металл: {'box':(x,y,w,h), 'click':(cx,cy), 'area':int,
                    'extent':float, 'snow_ring':float, 'type':'metal'}.

    Логика: тот же пайплайн что detect_stones, но:
      - маска цвета: METAL_LOWER/METAL_UPPER (серый, схоже с камнем)
      - проверка окружения: СНЕГ вместо травы (SNOW_RING_MIN)
    ВАЖНО: HSV металла требует калибровки на реальном скрине снежного биома!
    """
    H, W = bgr.shape[:2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    metal = cv2.inRange(hsv, np.array(METAL_LOWER), np.array(METAL_UPPER))
    snow  = cv2.inRange(hsv, np.array(SNOW_LOWER),  np.array(SNOW_UPPER))

    k = np.ones((MORPH_KERNEL, MORPH_KERNEL), np.uint8)
    metal = cv2.morphologyEx(metal, cv2.MORPH_CLOSE, k, iterations=STONE_CLOSE_ITERS)
    metal = cv2.morphologyEx(metal, cv2.MORPH_OPEN,  k, iterations=STONE_OPEN_ITERS)

    vchan = hsv[:, :, 2]
    schan = hsv[:, :, 1]

    contours, _ = cv2.findContours(metal, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    metals = []
    rejected = []

    for c in contours:
        area = cv2.contourArea(c)
        x, y, w, h = cv2.boundingRect(c)

        if area < MIN_METAL_AREA:
            continue
        if area > MAX_METAL_AREA:
            rejected.append(((x, y, w, h), "size"))
            continue

        extent = area / float(w * h) if w * h > 0 else 0.0
        aspect = max(w / float(h), h / float(w)) if w > 0 and h > 0 else 99
        if extent < MIN_EXTENT or aspect > MAX_ASPECT:
            rejected.append(((x, y, w, h), "shape"))
            continue

        cx, cy = x + w // 2, y + h // 2
        if _in_ignore_zone(cx, cy, W, H):
            rejected.append(((x, y, w, h), "ui"))
            continue

        sring = _snow_ring_ratio(snow, (x, y, w, h), W, H)
        if sring < SNOW_RING_MIN:
            rejected.append(((x, y, w, h), "no_snow"))
            continue

        dring = _dark_ring_ratio(vchan, schan, (x, y, w, h), W, H)
        if dring > DARK_RING_MAX and sring < DARK_SNOW_MIN:
            rejected.append(((x, y, w, h), "near_building"))
            continue

        roi_mask = metal[y:y + h, x:x + w]
        roi_v = vchan[y:y + h, x:x + w][roi_mask > 0]
        roi_s = schan[y:y + h, x:x + w][roi_mask > 0]
        vstd = float(np.std(roi_v)) if roi_v.size >= 5 else 0.0
        sstd = float(np.std(roi_s)) if roi_s.size >= 5 else 0.0
        if vstd < STONE_VSTD_MIN and sstd < STONE_SSTD_MIN:
            rejected.append(((x, y, w, h), "flat"))
            continue

        metals.append({
            'box':       (x, y, w, h),
            'click':     (cx, cy),
            'area':      int(area),
            'extent':    round(extent, 2),
            'snow_ring': round(sring, 2),
            'vstd':      round(vstd, 1),
            'sstd':      round(sstd, 1),
            'type':      'metal',
        })

    return metals, rejected, metal, snow


# --- Сигнал добычи камня ---
# Замер (diag_capture в живой игре): зелёного ГЛОУ у камня НЕТ (в отличие от
# дерева). Сигнал ИНВЕРТИРОВАН: перс приходит и ВСТАЁТ на камень -> заслоняет
# его телом -> пиксели КАМНЯ в зоне падают; добыл и отошёл -> пиксели вернулись.
#   камень_px высокий -> перс идёт (камень виден)
#   камень_px упал     -> перс пришёл, ДОБЫВАЕТ (заслонил)
#   камень_px вернулся -> ДОБЫТО
# Поэтому следим за stone_px (кол-во пикселей цвета камня в зоне).

def stone_px(bgr, cx, cy, r):
    """Кол-во пикселей цвета КАМНЯ в зоне радиусом r вокруг (cx,cy)."""
    H, W = bgr.shape[:2]
    x1, x2 = max(0, cx - r), min(W, cx + r)
    y1, y2 = max(0, cy - r), min(H, cy + r)
    if x2 <= x1 or y2 <= y1:
        return 0
    hsv = cv2.cvtColor(bgr[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, np.array(STONE_LOWER), np.array(STONE_UPPER))
    return int(np.count_nonzero(m))


def metal_px(bgr, cx, cy, r):
    """Кол-во пикселей цвета МЕТАЛЛА в зоне радиусом r вокруг (cx,cy).
    Сигнал добычи — тот же инвертированный принцип что у камня (заслон телом).
    """
    H, W = bgr.shape[:2]
    x1, x2 = max(0, cx - r), min(W, cx + r)
    y1, y2 = max(0, cy - r), min(H, cy + r)
    if x2 <= x1 or y2 <= y1:
        return 0
    hsv = cv2.cvtColor(bgr[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, np.array(METAL_LOWER), np.array(METAL_UPPER))
    return int(np.count_nonzero(m))


# Зелёный глоу (как у дерева) ОСТАВЛЕН для diag — у камня = шум/фон, не сигнал.
HIGHLIGHT_LOWER = (35, 40, 195)
HIGHLIGHT_UPPER = (90, 255, 255)


def highlight_count(bgr, cx, cy, r):
    """Кол-во пикселей яркого зелёного глоу в зоне (для diag; у камня не сигнал)."""
    H, W = bgr.shape[:2]
    x1, x2 = max(0, cx - r), min(W, cx + r)
    y1, y2 = max(0, cy - r), min(H, cy + r)
    if x2 <= x1 or y2 <= y1:
        return 0
    hsv = cv2.cvtColor(bgr[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
    neon = cv2.inRange(hsv, np.array(HIGHLIGHT_LOWER), np.array(HIGHLIGHT_UPPER))
    return int(np.count_nonzero(neon))
