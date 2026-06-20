# -*- coding: utf-8 -*-
"""
diag_glow.py
============
Диагностика подсветки. Находит дерево, кликает, и ~20с логирует глоу В ТОЧКЕ
ДЕРЕВА (камера статична) каждые 0.3с. Цель — замерить пороги: base (фон), пик
(во время руба), спад (когда срублено) -> NEON_RISE_PX / NEON_FALL_PX в bot.py.

Также логируем где сейчас МАКС глоу на экране — контроль что глоу там где
дерево (а не уехал/ложный).

Запуск:
  python diag_glow.py
Переключись в игру за 3 сек. Потом пришли весь вывод.
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
from tree_detector import (detect_trees, highlight_count,
                           HIGHLIGHT_LOWER, HIGHLIGHT_UPPER)


def max_glow_location(bgr):
    """Где на экране самый яркий глоу + его 'сила' (для детекта скролла камеры)."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    neon = cv2.inRange(hsv, np.array(HIGHLIGHT_LOWER), np.array(HIGHLIGHT_UPPER))
    n = int(np.count_nonzero(neon))
    if n == 0:
        return None, 0
    ys, xs = np.nonzero(neon)
    return (int(xs.mean()), int(ys.mean())), n


def main():
    print("Диагностика глоу. Старт через 3с — переключись в игру...")
    time.sleep(3)

    bgr, mon = bot.grab_screen()
    H, W = bgr.shape[:2]
    anchor = (int(bot.CHAR_ANCHOR[0] * W), int(bot.CHAR_ANCHOR[1] * H))
    trees, _, _, _ = detect_trees(bgr)
    trees = bot.pick_confident(trees)
    if not trees:
        print("Деревьев нет — наведи камеру на деревья и перезапусти.")
        return
    target = bot.choose_target(trees, None, anchor, [], [])
    cx, cy = target['click']
    print(f"Цель: дерево@({cx},{cy}). Кликаю...")

    win_input.click_abs(cx + mon['left'], cy + mon['top'])

    print(f"Слежу за глоу в точке дерева (r={bot.TREE_GLOW_R}), экран {W}x{H}")
    t0 = time.time()
    while time.time() - t0 < 20:
        bgr, _ = bot.grab_screen()
        at_tree = highlight_count(bgr, cx, cy, bot.TREE_GLOW_R)
        loc, total = max_glow_location(bgr)
        if loc:
            extra = f" | maxGlow@{loc} totalNeon={total}"
        else:
            extra = " | глоу на экране НЕТ"
        print(f"t={time.time()-t0:4.1f}s  neon_у_дерева={at_tree:5d}{extra}")
        time.sleep(0.3)

    print("Готово. Пришли весь вывод.")


if __name__ == "__main__":
    main()
