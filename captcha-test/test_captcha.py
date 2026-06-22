# -*- coding: utf-8 -*-
"""
test_captcha.py
===============
Тесты решения капчи «Verify you're human».

Правильные ответы задаются в TEST_CASES. Скрипт запускает несколько
методов предобработки и показывает pass/fail + итоговую точность.

Стратегия 2 попыток:
  Попытка 1 — лучший метод (определяется тестами).
  Попытка 2 — запасной метод если первый не совпал.
  Бот сам не знает правильный ответ — логика 2 попыток в нём: увидел новую
  капчу после Submit → решить снова другим методом.

Методы:
  RAW    — ddddocr на полном кадре
  BOX    — найти белый бокс → INTER_NEAREST ×3 → ddddocr
  BIN    — BOX + Otsu threshold → ddddocr  (линии стали чёрными — плохо)
  SAT    — BOX + saturation-маска (убрать серые линии) → ddddocr  (новый)

Запуск:
  cd captcha-test
  python test_captcha.py
"""

import os
import glob
import cv2
import numpy as np

# ── Правильные ответы (case-insensitive сравнение)
TEST_CASES = {
    "1.png": "9qs26",
    "2.png": "24trh",
}

CHAR_ZONE  = (0.10, 0.25, 0.75, 0.65)
UPSCALE    = 3
SAT_THRESH = 35   # saturation > этого = символ (не серая линия/фон)

# Замены спутанных символов: что ddddocr читает → что вероятно имелось в виду.
# Применяются на попытку 2 когда попытка 1 отклонена игрой.
CONFUSABLES = {
    '0': 'Q',   # округлая форма, Q без хвоста → 0
    '5': 'S',   # S в пиксельном шрифте → 5
    'F': '6',   # 6 с засечкой → F
    'O': 'Q',   # O и Q одинаковы в пикселях
    '1': 'I',   # тонкая вертикаль → 1 или I
    'Z': '2',   # Z и 2 похожи
}

SAVE_DEBUG = True


# ── ddddocr
_ocr = None
def get_ocr():
    global _ocr
    if _ocr is None:
        import ddddocr
        _ocr = ddddocr.DdddOcr(show_ad=False)
    return _ocr

def ocr(img_bgr):
    _, buf = cv2.imencode('.png', img_bgr)
    return get_ocr().classification(buf.tobytes()).upper().strip()


# ── Предобработки

def crop_zone(img):
    h, w = img.shape[:2]
    x0, y0 = int(w * CHAR_ZONE[0]), int(h * CHAR_ZONE[1])
    x1, y1 = int(w * CHAR_ZONE[2]), int(h * CHAR_ZONE[3])
    return img[y0:y1, x0:x1], (x0, y0, x1, y1)


def find_white_box(img_bgr):
    """Найти белый прямоугольник капчи в кропе."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 210, 255, cv2.THRESH_BINARY)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, k)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = [c for c in contours
                  if cv2.boundingRect(c)[2] > 40 and cv2.boundingRect(c)[3] > 20]
    if not candidates:
        return img_bgr, None
    x, y, w, h = cv2.boundingRect(max(candidates, key=cv2.contourArea))
    pad = 4
    x, y = max(0, x+pad), max(0, y+pad)
    w, h = min(img_bgr.shape[1]-x, w-2*pad), min(img_bgr.shape[0]-y, h-2*pad)
    return img_bgr[y:y+h, x:x+w], (x, y, x+w, y+h)


def nearest_upscale(img, factor):
    h, w = img.shape[:2]
    return cv2.resize(img, (w*factor, h*factor), interpolation=cv2.INTER_NEAREST)


def apply_bin(img_bgr):
    """Otsu threshold — линии становятся чёрными (плохо, но тестируем)."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, t = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.cvtColor(t, cv2.COLOR_GRAY2BGR)


def apply_sat(img_bgr):
    """
    Saturation-маска: серые линии (S≈0) → белый фон, цветные символы → чёрный.
    Линии и фон оба серые → выпадают. Символы любого цвета → остаются.
    """
    hsv  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask = (hsv[:, :, 1] > SAT_THRESH).astype(np.uint8) * 255
    # небольшой дилейт чтоб символы не рвались
    k    = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    mask = cv2.dilate(mask, k, iterations=1)
    # чёрные символы на белом фоне
    result = np.full_like(img_bgr, 255)
    result[mask > 0] = [0, 0, 0]
    return result


# ── Пост-обработка текста

def apply_substitutions(text):
    """Заменить спутанные символы (попытка 2)."""
    return ''.join(CONFUSABLES.get(c, c) for c in text.upper())


# ── Методы: (название, img_func, text_post_func или None)
# img_func:      image → image (предобработка перед OCR)
# text_post_func: text → text  (пост-обработка результата OCR), None = нет

def _box_img(img):
    crop, _ = crop_zone(img)
    box,  _ = find_white_box(crop)
    return nearest_upscale(box, UPSCALE)

METHODS = [
    ("RAW",   lambda img: img,                          None),
    ("BOX",   _box_img,                                 None),
    ("BIN",   lambda img: apply_bin(_box_img(img)),     None),
    ("SAT",   lambda img: apply_sat(_box_img(img)),     None),
    ("BOX+S", _box_img,                                 apply_substitutions),
]


# ── Тест

def check(got, expected):
    """Pass если совпадают по всем символам (case-insensitive)."""
    return got.lower() == expected.lower()


def run_tests():
    images_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")
    # только оригиналы (не debug-файлы)
    skip = ("_crop","_box","_bin","_sat","_raw","_big","_zones","_zone","_hsv")
    all_files = sorted(glob.glob(os.path.join(images_dir, "*.png")) +
                       glob.glob(os.path.join(images_dir, "*.jpg")))
    files = [f for f in all_files if not any(s in os.path.basename(f) for s in skip)]

    if not files:
        print(f"Нет картинок в {images_dir}")
        return

    # results[method_name] = [passed, total]
    totals = {name: [0, 0] for name, _, _ in METHODS}

    for path in files:
        img      = cv2.imread(path)
        name     = os.path.basename(path)
        expected = TEST_CASES.get(name)

        print(f"\n{'─'*52}")
        print(f"  {name}  ({img.shape[1]}×{img.shape[0]})")
        if expected:
            print(f"  Правильный ответ: «{expected}»")
        else:
            print(f"  Правильный ответ: неизвестен")

        base = os.path.splitext(path)[0]

        for mname, img_func, text_post in METHODS:
            processed = img_func(img)
            raw_text  = ocr(processed)
            result    = text_post(raw_text) if text_post else raw_text
            passed    = check(result, expected) if expected else None

            # показываем что было до и после замены
            if text_post and raw_text != result:
                label = f"«{raw_text}» → «{result}»"
            else:
                label = f"«{result}»"

            status = ("✓ PASS" if passed is True
                      else "✗ FAIL" if passed is False
                      else "?")

            print(f"  {mname:<6} {label:<28} {status}")

            if expected:
                totals[mname][1] += 1
                if passed:
                    totals[mname][0] += 1

            if SAVE_DEBUG and mname not in ("BOX+S",):
                cv2.imwrite(f"{base}_{mname.lower()}.png", processed)

    # Итог
    print(f"\n{'═'*52}")
    print("  ИТОГ:")
    for mname, (passed, total) in totals.items():
        pct = f"{passed/total*100:.0f}%" if total else "—"
        bar = "█" * passed + "░" * (total - passed)
        print(f"  {mname:<6} {passed}/{total}  {pct:>4}  {bar}")
    print()

    # Рекомендация: попытка 1 = без замен, попытка 2 = с заменами
    box_ok  = totals.get("BOX",   [0,1])
    subst_ok = totals.get("BOX+S", [0,1])
    combined = box_ok[0] + max(subst_ok[0] - box_ok[0], 0)
    total    = box_ok[1]
    print(f"  Стратегия 2 попыток (BOX → BOX+S):")
    print(f"    Покрытие: {combined}/{total} капч решается хотя бы одним методом")


if __name__ == "__main__":
    run_tests()
