# -*- coding: utf-8 -*-
"""
tool_select.py
==============
Вибір інструменту після входу в ігровий світ.

Алгоритм:
  1. Шукаємо цифри «1»–«6» у нижній частині екрану через OCR.
     Це мітки слотів хотбара — вони унікально знаходяться внизу.
  2. Від кожної знайденої цифри піднімаємося на SLOT_ICON_DY_FRAC угору —
     там іконка предмета.
  3. Наводимо мишу → tooltip з'являється → знову OCR → шукаємо назву.
  4. Збіг з target_tool → клік → повертаємо True.
  Fallback: якщо цифри не знайдено (OCR не спрацював) — використовуємо
  фіксовані фракції з CONFIG бота.
"""

import re
import time

# Частка висоти екрана: підняти від мітки-цифри до центру іконки.
# На скрині слот ~65px, цифра внизу → іконка ~35px вище.
# За замовчуванням 0.05 (підбери під своє розрішення якщо не попадає).
SLOT_ICON_DY_FRAC = 0.07

# Нижня межа (частка висоти) для пошуку цифр — нижче цього = хотбар.
# Відсіює цифри рівнів гравців / чату.
HOTBAR_MIN_Y_FRAC = 0.80

# Мін. впевненість OCR для цифри-мітки слоту.
DIGIT_CONF = 0.35

# Пауза після наведення перед читанням tooltip, сек.
HOVER_SETTLE = 0.5

# Мін. впевненість OCR для тексту tooltip.
TOOLTIP_CONF = 0.40

# --- Дефолтні фіксовані позиції (fallback якщо цифри не знайдено) ---
DEFAULT_SLOT_X_FRACS = [0.40, 0.44, 0.49, 0.53, 0.57, 0.62]
DEFAULT_SLOT_Y_FRAC  = 0.88   # центр іконок хотбара (цифри нижче ~0.96)


# ---------------------------------------------------------------------------

def _find_hotbar_slots(bgr, engine):
    """
    Знайти центри іконок слотів за цифрами «1»–«6» у нижній частині кадру.
    Повертає список (cx, cy_icon) відсортований за номером слоту,
    або [] якщо знайдено < 2 цифр.
    """
    from reconnect import _run_ocr
    H, W = bgr.shape[:2]
    dy = int(SLOT_ICON_DY_FRAC * H)
    min_y = int(HOTBAR_MIN_Y_FRAC * H)

    items = _run_ocr(engine, bgr)
    slots = {}

    for text, (cx, cy), conf in items:
        t = text.strip()
        if t not in ("1", "2", "3", "4", "5", "6"):
            continue
        if conf < DIGIT_CONF:
            continue
        if cy < min_y:          # фільтр: тільки нижня частина (хотбар)
            continue
        n = int(t)
        # Якщо та сама цифра знайдена двічі — беремо з вищою впевненістю
        if n not in slots or conf > slots[n][2]:
            slots[n] = (cx, cy - dy, conf)   # іконка — вище цифри

    if len(slots) < 2:
        return []

    return [(cx, cy) for n, (cx, cy, _) in sorted(slots.items())]


def select_tool(region, target_tool, templates, grab_fn,
                slot_x_fracs=None, slot_y_frac=None,
                dry_run=False, log=print):
    """
    Вибрати інструмент у хотбарі за tooltip-назвою.

    region      — dict left/top/width/height (абс. координати).
    target_tool — "Axe" або "Pickaxe". Без урахування регістру.
    templates   — reconnect-шаблоны (для OCR-рушія).
    grab_fn     — callable() -> BGR кадру регіону.
    slot_x_fracs / slot_y_frac — фіксовані фракції (fallback).
    dry_run     — True = тільки логи, без кліків.
    log         — функція логування.
    Повертає True якщо вибрано, False якщо не знайдено.
    """
    from reconnect import _get_ocr, _run_ocr
    import win_input

    engine = (templates.get('_ocr') if isinstance(templates, dict)
              else _get_ocr())
    if engine is None:
        log("  [tool] OCR недоступний — вибір інструменту пропущено.")
        return False

    target_up = target_tool.upper()
    W = region['width']
    H = region['height']

    log(f"  [tool] Шукаю {target_tool} в хотбарі...")

    # --- 1. Виявлення слотів за цифрами ---
    bgr0 = grab_fn()
    detected = _find_hotbar_slots(bgr0, engine)

    if detected:
        # X з цифр точний; Y беремо з CONFIG — стабільніше ніж offset від glyph
        yf = slot_y_frac if slot_y_frac is not None else DEFAULT_SLOT_Y_FRAC
        icon_y_abs = region['top'] + int(yf * H)
        positions = [(region['left'] + cx, icon_y_abs) for cx, _ in detected]
        log(f"  [tool] Знайдено {len(detected)} слотів (X авто, Y={yf}).")
    else:
        # Fallback: фіксовані фракції з CONFIG
        xs = slot_x_fracs if slot_x_fracs is not None else DEFAULT_SLOT_X_FRACS
        yf = slot_y_frac  if slot_y_frac  is not None else DEFAULT_SLOT_Y_FRAC
        cy_abs = region['top'] + int(yf * H)
        positions = [(region['left'] + int(f * W), cy_abs) for f in xs]
        log(f"  [tool] Цифри не знайдено — використовую фіксовані позиції ({len(positions)} слотів).")

    # --- 2. Hover → tooltip → click ---
    for i, (ax, ay) in enumerate(positions, start=1):
        if not dry_run:
            win_input.move_abs(ax, ay)
            time.sleep(HOVER_SETTLE)
        else:
            time.sleep(0.05)

        bgr = grab_fn()
        # OCR без фільтрації зон — tooltip знаходиться в зоні хотбара,
        # яку reconnect ігнорує, але нам вона потрібна
        items = _run_ocr(engine, bgr)

        for text, center, conf in items:
            words = re.split(r'\W+', text.upper())
            if conf >= TOOLTIP_CONF and target_up in words:
                log(f"  [tool] {target_tool} знайдено в слоті {i} — клік.")
                if not dry_run:
                    win_input.click_abs(ax, ay)
                    time.sleep(0.2)
                return True

    log(f"  [tool] {target_tool} не знайдено в {len(positions)} слотах.")
    return False
