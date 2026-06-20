# -*- coding: utf-8 -*-
"""
diag_glow.py (КАМЕНЬ)
=====================
Диагностика подсветки добычи камня. Находит камень, кликает, и ~20с логирует
глоу В ТОЧКЕ КАМНЯ (камера статична) каждые 0.3с. Цель — замерить пороги:
  base  — глоу до прихода перса (фон)
  пик   — сколько px глоу во время добычи
  спад  — до чего падает когда добыто
По этим числам выставить NEON_RISE_PX / NEON_FALL_PX в bot.py (сейчас копия
из дерева, непроверена).

Запуск:
  python diag_glow.py
Переключись в игру за 3 сек, наведи камеру на камень. Пришли весь вывод.
"""

import os, sys
import time
import numpy as np
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "common"))   # общий win_input
import win_input
win_input.set_dpi_aware()

import bot
from stone_detector import (detect_stones, highlight_count,
                            HIGHLIGHT_LOWER, HIGHLIGHT_UPPER)


def max_glow_location(bgr):
    """Где на экране самый яркий глоу + его 'сила' (контроль: глоу там где камень)."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    neon = cv2.inRange(hsv, np.array(HIGHLIGHT_LOWER), np.array(HIGHLIGHT_UPPER))
    n = int(np.count_nonzero(neon))
    if n == 0:
        return None, 0
    ys, xs = np.nonzero(neon)
    return (int(xs.mean()), int(ys.mean())), n


def main():
    print("Диагностика глоу добычи камня. Старт через 3с — переключись в игру...")
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

    win_input.click_abs(cx + mon['left'], cy + mon['top'])

    print(f"Слежу за глоу в точке камня (r={bot.STONE_GLOW_R}), экран {W}x{H}")
    t0 = time.time()
    while time.time() - t0 < 20:
        bgr, _ = bot.grab_screen()
        at_stone = highlight_count(bgr, cx, cy, bot.STONE_GLOW_R)
        loc, total = max_glow_location(bgr)
        extra = f" | maxGlow@{loc} totalNeon={total}" if loc else " | глоу на экране НЕТ"
        print(f"t={time.time()-t0:4.1f}s  neon_у_камня={at_stone:5d}{extra}")
        time.sleep(0.3)

    print("Готово. Пришли весь вывод.")


if __name__ == "__main__":
    main()
