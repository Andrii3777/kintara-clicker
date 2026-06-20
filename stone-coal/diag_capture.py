# -*- coding: utf-8 -*-
"""
diag_capture.py (КАМЕНЬ)
========================
Зелёного глоу у камня НЕТ (см. diag_glow). Надо УВИДЕТЬ реальный сигнал добычи.
Скрипт: находит камень, кликает, и ~14с каждые 0.5с:
  - меряет пиксели КАМНЯ (stone-маска) в зоне у точки клика — гипотеза:
    камень добыт -> исчезает -> пиксели падают.
  - меряет зелёный глоу там же (контроль, должен быть ~0).
  - сохраняет кадры cap_t<сек>.png на ключевых моментах — глянуть глазами что
    за анимация/подсветка (трещина, вспышка, исчезновение).

Запуск:
  python diag_capture.py
Наведи камеру на камень, переключись за 3с. Пришли вывод + кадры cap_*.png.
"""

import os, sys
import time
import numpy as np
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "common"))
import win_input
win_input.set_dpi_aware()

import bot
from stone_detector import (detect_stones, highlight_count,
                            STONE_LOWER, STONE_UPPER)

R = 50          # радиус зоны у камня
SAVE_AT = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 13.0]  # когда сохранять кадры


def stone_px(bgr, cx, cy, r):
    """Пиксели цвета КАМНЯ в зоне (исчез -> упадёт)."""
    H, W = bgr.shape[:2]
    x1, x2 = max(0, cx - r), min(W, cx + r)
    y1, y2 = max(0, cy - r), min(H, cy + r)
    hsv = cv2.cvtColor(bgr[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, np.array(STONE_LOWER), np.array(STONE_UPPER))
    return int(np.count_nonzero(m))


def main():
    print("Захват сигнала добычи камня. Старт через 3с — переключись в игру...")
    time.sleep(3)

    bgr, mon = bot.grab_screen()
    H, W = bgr.shape[:2]
    anchor = (int(bot.CHAR_ANCHOR[0] * W), int(bot.CHAR_ANCHOR[1] * H))
    stones, _, _, _ = detect_stones(bgr)
    stones = bot.pick_confident(stones)
    if not stones:
        print("Камней нет — наведи камеру на камни и перезапусти.")
        return
    target = bot.choose_target(stones, None, anchor, [], [])
    cx, cy = target['click']
    print(f"Цель: камень@({cx},{cy}) area={target['area']}. Кликаю...")
    base_stone = stone_px(bgr, cx, cy, R)
    print(f"Камень-пиксели ДО: {base_stone} (зона r={R})")

    win_input.click_abs(cx + mon['left'], cy + mon['top'])

    saved = set()
    t0 = time.time()
    while True:
        el = time.time() - t0
        if el > 14:
            break
        bgr, _ = bot.grab_screen()
        sp = stone_px(bgr, cx, cy, R)
        gl = highlight_count(bgr, cx, cy, R)
        print(f"t={el:4.1f}s  камень_px={sp:5d}  зелён_глоу={gl:5d}")
        for mark in SAVE_AT:
            if mark not in saved and el >= mark:
                # вырезаем зону вокруг камня + чуть контекста
                x1, y1 = max(0, cx - 120), max(0, cy - 120)
                x2, y2 = min(W, cx + 120), min(H, cy + 120)
                cv2.imwrite(f"cap_t{int(mark)}.png", bgr[y1:y2, x1:x2])
                saved.add(mark)
        time.sleep(0.5)

    print("Готово. Пришли вывод + кадры cap_*.png.")


if __name__ == "__main__":
    main()
