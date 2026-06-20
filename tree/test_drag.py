# -*- coding: utf-8 -*-
"""
test_drag.py
============
Тест панорамирования карты перетягиванием. Делает ОДИН drag и сохраняет
before.png / after.png — глазами проверить что карта сдвинулась.

Запуск:
  python test_drag.py                  # вправо на 500px, монитор из bot.py
  python test_drag.py --dir left
  python test_drag.py --dir up --dist 700 --monitor 1

Направление = куда ТЯНЕМ мышь. Карта поедет в ту же сторону, вид — в обратную.
Старт из центра экрана (безопасно, не задеть UI).
"""

import argparse
import os, sys
import time
import numpy as np
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "common"))   # общий win_input
import win_input
win_input.set_dpi_aware()


def grab(monitor_index):
    import mss
    with mss.mss() as sct:
        mon = sct.monitors[monitor_index]
        shot = np.array(sct.grab(mon))
        return cv2.cvtColor(shot, cv2.COLOR_BGRA2BGR), mon


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="SE",
                    choices=["left", "right", "up", "down",
                             "NE", "SW", "NW", "SE"],
                    help="орто или диагональ (NE=↗ SW=↙ NW=↖ SE=↘)")
    ap.add_argument("--dist", type=int, default=380, help="длина по оси, px")
    ap.add_argument("--monitor", type=int, default=None,
                    help="номер монитора (по умолч из bot.MONITOR_INDEX)")
    ap.add_argument("--duration", type=float, default=0.6, help="время drag, сек")
    args = ap.parse_args()

    if args.monitor is None:
        import bot
        mon_idx = bot.MONITOR_INDEX
    else:
        mon_idx = args.monitor

    print(f"Тест drag: dir={args.dir} dist={args.dist}px монитор[{mon_idx}]")
    print("Старт через 3с — переключись в игру...")
    time.sleep(3)

    before, mon = grab(mon_idx)
    H, W = before.shape[:2]
    cv2.imwrite("before.png", before)

    # старт из центра монитора (в координатах виртуального стола)
    sx = mon['left'] + W // 2
    sy = mon['top'] + H // 2
    vec = {"left": (-1, 0), "right": (1, 0), "up": (0, -1), "down": (0, 1),
           "NE": (1, -1), "SW": (-1, 1), "NW": (-1, -1), "SE": (1, 1)}[args.dir]
    dx, dy = vec[0] * args.dist, vec[1] * args.dist
    ex, ey = sx + dx, sy + dy

    print(f"drag ({sx},{sy}) -> ({ex},{ey})")
    win_input.drag_abs(sx, sy, ex, ey, steps=25, duration=args.duration)

    time.sleep(0.5)   # дать карте устаканиться
    after, _ = grab(mon_idx)
    cv2.imwrite("after.png", after)

    # насколько сильно изменился кадр (грубая мера что что-то сдвинулось)
    diff = cv2.absdiff(before, after)
    changed = float(np.count_nonzero(diff.sum(2) > 30)) / (H * W)
    print(f"Сохранил before.png / after.png. Изменилось пикселей: {changed*100:.1f}%")
    if changed < 0.05:
        print("МАЛО изменений — карта почти не сдвинулась. Возможно drag не сработал.")
    else:
        print("Карта сдвинулась. Сравни before.png / after.png.")


if __name__ == "__main__":
    main()
