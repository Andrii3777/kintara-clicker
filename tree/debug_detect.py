# -*- coding: utf-8 -*-
"""
debug_detect.py
===============
Тестовый прогон детекции. БЕЗ кликов.
Берёт картинку (файл ИЛИ скрин экрана), запускает detect_trees,
рисует рамки и сохраняет результат. Смотрим глазами что поймалось.

Запуск:
  python debug_detect.py --image assets/screen.png  # на готовом эталон-кадре
  python debug_detect.py --screen               # захват своего экрана

Что рисует на debug_result.png:
  ЗЕЛЁНАЯ рамка  = дерево (засчитано), синяя точка = куда кликнули бы
  КРАСНАЯ рамка  = коричневое пятно отклонено (size / ui / no_crown)

Дополнительно сохраняет:
  debug_green_mask.png  — что поймала зелёная маска
  debug_brown_mask.png  — что поймала коричневая маска
Эти 2 файла — главный инструмент чтоб крутить HSV в tree_detector.py.
"""

import argparse
import cv2
import numpy as np

from tree_detector import detect_trees


def grab_screen(monitor_index=1):
    """Захват выбранного монитора -> BGR. Через mss (быстро)."""
    import mss
    with mss.mss() as sct:
        mon = sct.monitors[monitor_index]   # [1]=первый, [2]=второй, [0]=все
        shot = np.array(sct.grab(mon))      # BGRA
        return cv2.cvtColor(shot, cv2.COLOR_BGRA2BGR)


def draw_debug(bgr, trees, rejected):
    """Рисуем результат поверх копии картинки."""
    out = bgr.copy()

    # отклонённые — красным, подписываем причину
    for (x, y, w, h), reason in rejected:
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 0, 255), 1)
        cv2.putText(out, reason, (x, y - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

    # деревья — зелёным + точка клика синим
    for t in trees:
        x, y, w, h = t['box']
        cx, cy = t['click']
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(out, (cx, cy), 5, (255, 0, 0), -1)
        cv2.putText(out, f"{t['crown_ratio']:.2f}", (x, y - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    return out


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--image", help="путь к картинке-скрину")
    g.add_argument("--screen", action="store_true", help="захватить свой экран")
    ap.add_argument("--monitor", type=int, default=1,
                    help="номер монитора: 1=первый, 2=второй, 0=все (по умолч 1)")
    args = ap.parse_args()

    if args.screen:
        print(f"Захват монитора [{args.monitor}] через 2 сек — переключись в игру...")
        import time
        time.sleep(2)
        bgr = grab_screen(args.monitor)
    else:
        bgr = cv2.imread(args.image)
        if bgr is None:
            print(f"Не смог открыть файл: {args.image}")
            return

    trees, rejected, green, brown = detect_trees(bgr)

    print(f"Деревьев найдено: {len(trees)}")
    print(f"Отклонено пятен: {len(rejected)}")

    out = draw_debug(bgr, trees, rejected)
    cv2.imwrite("debug_result.png", out)
    cv2.imwrite("debug_green_mask.png", green)
    cv2.imwrite("debug_brown_mask.png", brown)
    print("Сохранил: debug_result.png, debug_green_mask.png, debug_brown_mask.png")


if __name__ == "__main__":
    main()
