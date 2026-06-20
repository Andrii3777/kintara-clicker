# import ctypes
# import random
# import time
# import cv2
# import numpy as np
# import pyautogui
# from PIL import ImageGrab
#
# # Fix DPI scaling
# try:
#     ctypes.windll.shcore.SetProcessDpiAwareness(2)
# except Exception:
#     pass
#
# # --- Config ---
# # Расширенный фильтр для захвата самого темного угля
# HSV_LOWER = np.array([12, 24, 35])
# HSV_UPPER = np.array([35, 65, 130])
#
# MAX_DIST = 350
# MINE_DELAY = 7.5
# BLACKLIST_TIME = 25.0  # Сколько секунд бот будет игнорировать место добычи (панельку)
#
# # ВАЖНО: Впиши сюда точные координаты твоих 3 окон на мониторе (X1, Y1, X2, Y2)
# WINDOWS = [
#     {"name": "Window-1", "roi": (0, 0, 1280, 720), "next_run": 0, "blacklist": []},
#     {"name": "Window-2", "roi": (1280, 0, 2560, 720), "next_run": 0, "blacklist": []},
#     {"name": "Window-3", "roi": (0, 720, 1280, 1440), "next_run": 0, "blacklist": []}
# ]
#
# EASINGS = [
#     pyautogui.easeInQuad,
#     pyautogui.easeOutQuad,
#     pyautogui.easeInOutQuad,
#     pyautogui.linear
# ]
#
#
# def find_target(roi, blacklist):
#     try:
#         img = ImageGrab.grab(bbox=roi)
#         rgb = np.array(img)
#         hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
#         gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
#
#         mask = cv2.inRange(hsv, HSV_LOWER, HSV_UPPER)
#         kernel = np.ones((3, 3), np.uint8)
#         mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
#
#         contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
#         w = roi[2] - roi[0]
#         h = roi[3] - roi[1]
#         center_x = w // 2
#         center_y = h // 2
#
#         current_time = time.time()
#
#         targets = []
#         for cnt in contours:
#             area = cv2.contourArea(cnt)
#
#             # Площадь: от маленьких дальних камней до огромного угля
#             if not (60 < area < 3000):
#                 continue
#
#             bx, by, bw, bh = cv2.boundingRect(cnt)
#             ar = float(bw) / float(bh)
#
#             # Игнорируем длинные тонкие объекты (кирки и интерфейс)
#             if not (0.6 < ar < 2.5):
#                 continue
#
#             cx = bx + bw // 2
#             cy = by + bh // 2
#
#             # Мертвая зона: не кликаем по герою
#             if np.sqrt((cx - center_x) ** 2 + (cy - center_y) ** 2) < 110:
#                 continue
#
#             # === СИСТЕМА ПАМЯТИ (Игнор оставленных панелек) ===
#             is_rubble = False
#             for rx, ry, expire_time in blacklist:
#                 if current_time < expire_time:
#                     # Если этот объект в радиусе 60 пкс от недавней добычи - это мусор!
#                     if np.sqrt((cx - rx) ** 2 + (cy - ry) ** 2) < 60:
#                         is_rubble = True
#                         break
#             if is_rubble:
#                 continue
#
#             # Текстурный фильтр: убивает идеально гладкий пол, если он прошел по цвету
#             local_mask = np.zeros((bh, bw), dtype=np.uint8)
#             cv2.drawContours(local_mask, [cnt - [bx, by]], -1, 255, -1)
#             roi_gray = gray[by:by + bh, bx:bx + bw]
#             pixels = roi_gray[local_mask == 255]
#
#             if len(pixels) > 0:
#                 std_dev = np.var(pixels) ** 0.5
#                 if std_dev < 6.5:
#                     continue
#             else:
#                 continue
#
#             dist = np.sqrt((cx - center_x) ** 2 + (cy - center_y) ** 2)
#
#             # Приоритет на темное (Уголь)
#             mean_val = np.mean(pixels)
#             score = dist + (mean_val * 2)
#
#             if dist <= MAX_DIST:
#                 targets.append((cx, cy, score))
#
#         if not targets:
#             return None
#
#         targets.sort(key=lambda x: x[2])
#         best_cx, best_cy, _ = targets[0]
#
#         # Пересчет координат в глобальные координаты монитора
#         screen = ImageGrab.grab()
#         sx = pyautogui.size()[0] / screen.size[0]
#         sy = pyautogui.size()[1] / screen.size[1]
#
#         fx = int((roi[0] + best_cx) * sx) + random.randint(-3, 3)
#         fy = int((roi[1] + best_cy) * sy) + random.randint(-3, 3)
#
#         # Возвращаем глобальные клики И локальные центры для блэклиста
#         return fx, fy, best_cx, best_cy
#
#     except Exception as e:
#         print(f"Vision error: {e}")
#         return None
#
#
# print("Starting Multi-Box Bot in 5 seconds...")
# time.sleep(5)
#
# try:
#     while True:
#         now = time.time()
#
#         for w in WINDOWS:
#             if now < w["next_run"]:
#                 continue
#
#             # Очищаем старую память от уже исчезнувших панелек
#             w["blacklist"] = [item for item in w["blacklist"] if item[2] > now]
#
#             result = find_target(w["roi"], w["blacklist"])
#
#             if result:
#                 fx, fy, cx, cy = result
#                 print(f"[{w['name']}] Target found. Clicking ({fx}, {fy})")
#                 pyautogui.moveTo(fx, fy, duration=random.uniform(0.2, 0.4), tween=random.choice(EASINGS))
#                 pyautogui.click(duration=random.uniform(0.04, 0.08))
#
#                 # ЗАПОМИНАЕМ МЕСТО! Игнорируем эту зону 25 секунд
#                 w["blacklist"].append((cx, cy, now + BLACKLIST_TIME))
#
#                 # Таймер добычи для текущего окна
#                 w["next_run"] = now + MINE_DELAY
#             else:
#                 w["next_run"] = now + 1.0
#
#         time.sleep(0.05)
#
# except KeyboardInterrupt:
#     print("\nBot stopped.")

import ctypes
import random
import time
import cv2
import numpy as np
import pyautogui
from PIL import ImageGrab
import os

# Фікс масштабування Windows
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
except Exception:
    pass

# ================= НАЛАШТУВАННЯ =================
HSV_LOWER = np.array([10, 20, 30])
HSV_UPPER = np.array([45, 110, 165])

MAX_DIST = 350
MINE_DELAY = 7.5
BLACKLIST_TIME = 25.0

WINDOWS = [
    {"name": "Ліве-Верхнє", "roi": (0, 0, 1280, 720), "next_ui": 0, "next_mine": 0, "blacklist": [],
     "target_server": "server_2", "empty_scans": 0},
    {"name": "Праве-Верхнє", "roi": (1280, 0, 2560, 720), "next_ui": 0, "next_mine": 0, "blacklist": [],
     "target_server": "server_3", "empty_scans": 0},
    {"name": "Ліве-Нижнє", "roi": (0, 720, 1280, 1440), "next_ui": 0, "next_mine": 0, "blacklist": [],
     "target_server": "server_4", "empty_scans": 0}
]

EASINGS = [pyautogui.easeInQuad, pyautogui.easeOutQuad, pyautogui.easeInOutQuad, pyautogui.linear]

print("Калібрування екрана...")
init_screen = ImageGrab.grab()
SX = pyautogui.size()[0] / init_screen.size[0]
SY = pyautogui.size()[1] / init_screen.size[1]
del init_screen

# ================= ЗАВАНТАЖЕННЯ ШАБЛОНІВ UI =================
UI_TEMPLATES = {}
template_names = ['ok', 'error', 'wait', 'server_list', 'server_2', 'server_3', 'server_4']

for name in template_names:
    path = f'ui/{name}.png'
    if os.path.exists(path):
        UI_TEMPLATES[name] = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    else:
        UI_TEMPLATES[name] = None


def check_ui_element(gray_img, template, threshold=0.92):
    if template is None: return None

    scales = [1.0, 0.95, 0.90, 0.85, 0.80, 1.05, 1.10]

    for scale in scales:
        width = int(template.shape[1] * scale)
        height = int(template.shape[0] * scale)

        if width < 10 or height < 10 or width > gray_img.shape[1] or height > gray_img.shape[0]:
            continue

        resized_template = cv2.resize(template, (width, height))
        res = cv2.matchTemplate(gray_img, resized_template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)

        if max_val >= threshold:
            return (max_loc[0] + width // 2, max_loc[1] + height // 2)

    return None


def emergency_ui_check(gray_img, target_server):
    if check_ui_element(gray_img, UI_TEMPLATES.get('wait'), threshold=0.95):
        return {"action": "wait", "msg": "Очікування запуску сервера..."}

    ok_pos = check_ui_element(gray_img, UI_TEMPLATES.get('ok'), threshold=0.90)
    if ok_pos:
        return {"action": "click", "x": ok_pos[0], "y": ok_pos[1], "msg": "Виліт! Натискаю ОК."}

    error_pos = check_ui_element(gray_img, UI_TEMPLATES.get('error'), threshold=0.90)
    if error_pos:
        return {"action": "click", "x": error_pos[0], "y": error_pos[1],
                "msg": "Connect Error! Закриваю вікно помилки."}

    # ПОВЕРНУТО ЖОРСТКИЙ ПОРІГ (0.93)
    server_pos = check_ui_element(gray_img, UI_TEMPLATES.get(target_server), threshold=0.93)
    if server_pos:
        return {"action": "click", "x": server_pos[0], "y": server_pos[1],
                "msg": f"Знайшов {target_server}! Заходжу на сервер."}

    # ПОВЕРНУТО ЖОРСТКИЙ ПОРІГ (0.90), щоб не плутав заголовок із кнопками
    list_pos = check_ui_element(gray_img, UI_TEMPLATES.get('server_list'), threshold=0.90)
    if list_pos:
        return {
            "action": "scroll",
            "x": list_pos[0],
            "y": list_pos[1],
            "msg": f"Сервер {target_server} не знайдено. Покроково зміщую список..."
        }

    return None


def find_target(window_dict, current_time):
    roi = window_dict["roi"]
    blacklist = window_dict["blacklist"]
    target_server = window_dict["target_server"]

    try:
        img = ImageGrab.grab(bbox=roi)
        rgb = np.array(img)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

        if current_time >= window_dict["next_ui"]:
            ui_action = emergency_ui_check(gray, target_server)
            if ui_action:
                return {"type": "ui", **ui_action}

        if current_time >= window_dict["next_mine"]:
            hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
            mask = cv2.inRange(hsv, HSV_LOWER, HSV_UPPER)
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            center_x, center_y = (roi[2] - roi[0]) // 2, (roi[3] - roi[1]) // 2
            targets = []

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if not (60 < area < 3000): continue

                bx, by, bw, bh = cv2.boundingRect(cnt)
                ar = float(bw) / float(bh)
                if not (0.6 < ar < 2.5): continue

                cx, cy = bx + bw // 2, by + bh // 2
                if np.sqrt((cx - center_x) ** 2 + (cy - center_y) ** 2) < 110: continue

                is_blacklisted = False
                for rx, ry, expire_time in blacklist:
                    if current_time < expire_time:
                        if np.sqrt((cx - rx) ** 2 + (cy - ry) ** 2) < 60:
                            is_blacklisted = True
                            break
                if is_blacklisted: continue

                local_mask = np.zeros((bh, bw), dtype=np.uint8)
                cv2.drawContours(local_mask, [cnt - [bx, by]], -1, 255, -1)
                roi_gray = gray[by:by + bh, bx:bx + bw]
                pixels = roi_gray[local_mask == 255]

                if len(pixels) > 0:
                    std_dev = np.var(pixels) ** 0.5
                    if std_dev < 7.3: continue
                else:
                    continue

                dist = np.sqrt((cx - center_x) ** 2 + (cy - center_y) ** 2)
                score = dist + (np.mean(pixels) * 2)
                if dist <= MAX_DIST:
                    targets.append((cx, cy, score))

            if not targets: return None
            targets.sort(key=lambda x: x[2])
            return {"type": "mine", "cx": targets[0][0], "cy": targets[0][1]}

    except Exception as e:
        print(f"Vision error: {e}")
        return None


print("=== Бот запущено: Жорсткі пороги та Безпечний фокус ===\n")
time.sleep(3)

try:
    while True:
        now = time.time()

        for w in WINDOWS:
            w["blacklist"] = [item for item in w["blacklist"] if item[2] > now]

            result = find_target(w, now)

            if result:
                w["empty_scans"] = 0

                if result["type"] == "ui":
                    print(f"[{w['name']}] {result['msg']}")

                    if result["action"] == "wait":
                        w["next_ui"] = now + 4.0
                        continue

                    fx = int((w["roi"][0] + result["x"]) * SX)
                    fy = int((w["roi"][1] + result["y"]) * SY)

                    if result["action"] == "click":
                        pyautogui.moveTo(fx, fy, duration=0.2)
                        pyautogui.click()
                        w["next_ui"] = now + 2.5

                    elif result["action"] == "scroll":
                        # БЕЗПЕЧНИЙ КЛІК: б'ємо на 15 пікселів ВИЩЕ центру заголовка
                        pyautogui.moveTo(fx, fy - 15, duration=0.2)
                        pyautogui.click()
                        time.sleep(0.1)

                        # Переміщуємо мишку безпосередньо в зону самого списку
                        pyautogui.moveTo(fx, fy + 150, duration=0.2)
                        time.sleep(0.1)

                        # Один крок коліщатком
                        pyautogui.scroll(-120)
                        w["next_ui"] = now + 1.8

                elif result["type"] == "mine":
                    cx, cy = result["cx"], result["cy"]
                    fx = int((w["roi"][0] + cx) * SX) + random.randint(-3, 3)
                    fy = int((w["roi"][1] + cy) * SY) + random.randint(-3, 3)

                    print(f"[{w['name']}] Копаю руду ({fx}, {fy})")
                    pyautogui.moveTo(fx, fy, duration=random.uniform(0.25, 0.45), tween=random.choice(EASINGS))
                    pyautogui.click(duration=random.uniform(0.04, 0.08))

                    w["blacklist"].append((cx, cy, now + BLACKLIST_TIME))
                    w["next_mine"] = now + MINE_DELAY
                    w["next_ui"] = now + 0.5
            else:
                if now >= w["next_ui"]: w["next_ui"] = now + 0.5
                if now >= w["next_mine"]: w["next_mine"] = now + 0.5

                w["empty_scans"] += 1
                if w["empty_scans"] == 30:
                    print(f"[{w['name']}] Спокійне сканування території...")
                    w["empty_scans"] = 0

        time.sleep(0.02)

except KeyboardInterrupt:
    print("\n[СТОП] Бот зупинений.")