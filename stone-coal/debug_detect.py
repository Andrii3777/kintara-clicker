# -*- coding: utf-8 -*-
"""
debug_detect.py (камень)
========================
Тест детекции камней БЕЗ кликов. Берёт картинку (файл/экран), запускает
detect_stones, рисует рамки и сохраняет результат. Смотрим глазами.

Запуск:
  python debug_detect.py --image images-examples/lots-of-stones.png
  python debug_detect.py --all          # прогнать ВСЕ примеры в images-examples/
  python debug_detect.py --screen --monitor 1

Рисует на <имя>_result.png:
  ЗЕЛЁНАЯ рамка = камень (точка = клик), КРАСНАЯ = отклонено (size/shape/ui/no_grass)
Плюс <имя>_stone_mask.png и <имя>_grass_mask.png — крутить HSV.
"""

import argparse
import glob
import os
import cv2
import numpy as np

from stone_detector import detect_stones


def grab_screen(monitor_index=1):
    import mss
    with mss.mss() as sct:
        mon = sct.monitors[monitor_index]
        shot = np.array(sct.grab(mon))
        return cv2.cvtColor(shot, cv2.COLOR_BGRA2BGR)


def draw_debug(bgr, stones, rejected):
    out = bgr.copy()
    for (x, y, w, h), reason in rejected:
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 0, 255), 1)
        cv2.putText(out, reason, (x, y - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
    for s in stones:
        x, y, w, h = s['box']
        cx, cy = s['click']
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(out, (cx, cy), 4, (255, 0, 0), -1)
        cv2.putText(out, f"{s['grass_ring']:.2f}", (x, y - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    return out


def run_on(bgr, stem):
    stones, rejected, stone, grass = detect_stones(bgr)
    print(f"[{stem}] камней: {len(stones)}  отклонено: {len(rejected)}")
    for s in stones:
        print(f"    {s['click']} area={s['area']} extent={s['extent']} "
              f"grass_ring={s['grass_ring']} vstd={s['vstd']} sstd={s['sstd']}")
    cv2.imwrite(f"{stem}_input.png", bgr)   # СЫРОЙ кадр — чтоб тюнить на нём же
    cv2.imwrite(f"{stem}_result.png", draw_debug(bgr, stones, rejected))
    cv2.imwrite(f"{stem}_stone_mask.png", stone)
    cv2.imwrite(f"{stem}_grass_mask.png", grass)


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--image", help="путь к картинке")
    g.add_argument("--screen", action="store_true", help="захватить экран")
    g.add_argument("--all", action="store_true", help="все примеры images-examples/")
    ap.add_argument("--monitor", type=int, default=1)
    args = ap.parse_args()

    if args.all:
        for p in sorted(glob.glob("images-examples/*.png")):
            bgr = cv2.imread(p)
            run_on(bgr, "dbg_" + os.path.splitext(os.path.basename(p))[0])
        return

    if args.screen:
        print(f"Захват монитора [{args.monitor}] через 2 сек...")
        import time
        time.sleep(2)
        bgr = grab_screen(args.monitor)
        run_on(bgr, "dbg_screen")
    else:
        bgr = cv2.imread(args.image)
        if bgr is None:
            print(f"Не открыл: {args.image}")
            return
        run_on(bgr, "dbg_" + os.path.splitext(os.path.basename(args.image))[0])


if __name__ == "__main__":
    main()
