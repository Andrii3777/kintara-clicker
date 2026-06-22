# -*- coding: utf-8 -*-
"""
captcha.py
==========
Капча «Verify you're human» (Kintara). Общий модуль для всех ботов.

Паттерн как reconnect.py: вход = BGR кадр, выход = dict действия или None.
Клик/ввод делает бот (win_input + keyboard). Тут — только зрение + решение.

Флоу:
  1. captcha_check(frame) — OCR ловит «VERIFY»+«HUMAN» или «TIME LEFT».
  2. Если капча — кропает зону символов (по OCR-якорям или фикс-долям).
  3. Решает: сначала Claude API (haiku — дёшево), фолбэк — HSV+RapidOCR.
  4. Возвращает {action:'captcha', text:'9Q526', input_x, input_y, submit_x, submit_y}.

Бот при получении 'captcha':
  click(input_x, input_y)  →  keyboard.write(text)  →  click(submit_x, submit_y)
  Координаты ЛОКАЛЬНЫЕ внутри региона (бот добавляет offset монитора/квадранта).
"""

import re
import cv2
import numpy as np

MIN_CONF = 0.45

# Фикс-доли кадра для зоны символов / кнопок (фолбэк если OCR не нашёл якоря).
# Калиброваны по скрину 554×442. Подгони если разрешение сильно другое.
CHAR_ZONE  = (0.13, 0.32, 0.68, 0.58)   # (x0, y0, x1, y1) доли кадра
INPUT_REL  = (0.40, 0.63)               # (cx, cy) доли кадра
SUBMIT_REL = (0.40, 0.86)              # (cx, cy) доли кадра

_OCR       = None
_OCR_FAIL  = False


# ---------------------------------------------------------------------------
# OCR (переиспользуем синглтон reconnect если уже загружен, иначе свой)
# ---------------------------------------------------------------------------
def _get_ocr():
    global _OCR, _OCR_FAIL
    if _OCR is not None or _OCR_FAIL:
        return _OCR
    try:
        from rapidocr_onnxruntime import RapidOCR
        _OCR = RapidOCR()
    except Exception as e:
        _OCR_FAIL = True
        print(f"  [captcha] RapidOCR недоступен ({e})")
    return _OCR


def _box_center(box):
    xs = [p[0] for p in box]
    ys = [p[1] for p in box]
    return (int(sum(xs) / 4), int(sum(ys) / 4))


def _run_ocr(frame):
    engine = _get_ocr()
    if engine is None:
        return []
    img = frame if frame.ndim == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    result, _ = engine(img)
    if not result:
        return []
    return [(text.upper().strip(), _box_center(box), float(conf))
            for box, text, conf in result]


def _find(items, *keywords):
    for text, center, conf in items:
        if conf >= MIN_CONF and any(k in text for k in keywords):
            return center
    return None


# ---------------------------------------------------------------------------
# Решатели
# ---------------------------------------------------------------------------
def _solve_with_claude(img_bgr):
    """Claude haiku — надёжнее всего. ~0.001$ за вызов."""
    try:
        import anthropic, base64
        _, buf = cv2.imencode('.png', img_bgr)
        b64 = base64.b64encode(buf.tobytes()).decode()
        client = anthropic.Anthropic()
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=20,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image",
                     "source": {"type": "base64",
                                "media_type": "image/png",
                                "data": b64}},
                    {"type": "text",
                     "text": ("What characters are shown in this CAPTCHA image? "
                              "Reply with ONLY the characters, no spaces, "
                              "no explanation, no punctuation.")}
                ]
            }]
        )
        raw = msg.content[0].text.strip().upper()
        clean = re.sub(r'[^A-Z0-9]', '', raw)
        print(f"  [captcha] Claude → «{clean}» (raw: {raw!r})")
        return clean or None
    except Exception as e:
        print(f"  [captcha] Claude API ошибка: {e}")
        return None


def _solve_with_ocr(img_bgr):
    """HSV-маска тёмно-фиолетовых символов → RapidOCR."""
    engine = _get_ocr()
    if engine is None:
        return None

    # Изолировать тёмно-фиолетовые/синие символы (убирает серые линии-помехи).
    # H 115-150 (синий/фиолетовый), S>80 (насыщенный), V<165 (тёмный).
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([115, 80, 30]), np.array([150, 255, 165]))

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    mask = cv2.dilate(mask, kernel, iterations=1)

    # Инверт: тёмное на светлом (OCR лучше читает так).
    clean = cv2.bitwise_not(mask)

    # Апскейл — кроп маленький, OCR нужны ~300+ px по ширине.
    h, w = clean.shape[:2]
    scale = max(1.0, 350.0 / max(w, 1))
    if scale > 1.0:
        clean = cv2.resize(clean, (int(w * scale), int(h * scale)),
                           interpolation=cv2.INTER_CUBIC)

    result, _ = engine(cv2.cvtColor(clean, cv2.COLOR_GRAY2BGR))
    if not result:
        print("  [captcha] OCR фолбэк: пусто")
        return None

    texts = [t for _, t, c in result if c >= 0.35]
    combined = re.sub(r'[^A-Z0-9]', '', "".join(texts).upper())
    print(f"  [captcha] OCR фолбэк → «{combined}»")
    return combined or None


def _solve_captcha(char_img):
    """Claude → OCR → None."""
    return _solve_with_claude(char_img) or _solve_with_ocr(char_img)


# ---------------------------------------------------------------------------
# Публичный API
# ---------------------------------------------------------------------------
def captcha_check(frame, templates=None):
    """
    Кадр (BGR) → dict действия капчи или None.

    Возвращаемый dict:
      {'action':'captcha', 'text':'9Q526',
       'input_x':int, 'input_y':int,   # центр поля ввода (локальные координаты)
       'submit_x':int, 'submit_y':int, # центр Submit (локальные координаты)
       'msg':str}

    templates — для совместимости с контрактом ботов (игнорируется, OCR свой).
    """
    items = _run_ocr(frame)
    if not items:
        return None

    H, W = frame.shape[:2]
    joined = " ".join(t for t, _, c in items if c >= MIN_CONF)

    # Детект: нужны оба слова вместе ИЛИ специфичная фраза таймера.
    has_captcha = (("VERIFY" in joined and "HUMAN" in joined)
                   or "TIME LEFT" in joined)
    if not has_captcha:
        return None

    # --- Локаторы элементов (OCR-якоря, фолбэк = фиксированные доли) ---
    submit_pos = _find(items, "SUBMIT")
    time_pos   = _find(items, "TIME LEFT", "TIME")
    verify_pos = _find(items, "VERIFY")

    # Submit
    sx = submit_pos[0] if submit_pos else int(W * SUBMIT_REL[0])
    sy = submit_pos[1] if submit_pos else int(H * SUBMIT_REL[1])

    # Input field — чуть выше "Time left" (примерно 1 строку)
    if time_pos:
        input_x = time_pos[0]
        input_y = time_pos[1] - int(H * 0.07)
    else:
        input_x = int(W * INPUT_REL[0])
        input_y = int(H * INPUT_REL[1])

    # Зона символов — между заголовком и полем ввода
    cx0 = int(W * CHAR_ZONE[0])
    cx1 = int(W * CHAR_ZONE[2])
    if verify_pos and time_pos:
        cy0 = verify_pos[1] + int(H * 0.08)   # ниже заголовка+описания
        cy1 = input_y       - int(H * 0.03)   # выше поля ввода
    else:
        cy0 = int(H * CHAR_ZONE[1])
        cy1 = int(H * CHAR_ZONE[3])

    cy0 = max(0, cy0)
    cy1 = min(H, cy1)
    cx0 = max(0, cx0)
    cx1 = min(W, cx1)

    if cy1 - cy0 < 20 or cx1 - cx0 < 20:
        print("  [captcha] Зона символов слишком мала, пропуск.")
        return None

    char_img = frame[cy0:cy1, cx0:cx1]

    # Отладка: сохранить кроп (раскомментировать при калибровке)
    # cv2.imwrite("captcha_crop.png", char_img)

    text = _solve_captcha(char_img)
    if not text:
        print("  [captcha] Не удалось решить капчу.")
        return None

    return {
        "action":   "captcha",
        "text":     text,
        "input_x":  input_x,
        "input_y":  input_y,
        "submit_x": sx,
        "submit_y": sy,
        "msg":      f"Капча — решение: «{text}»",
    }
