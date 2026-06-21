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
HSV_LOWER = np.array([5, 15, 20])
HSV_UPPER = np.array([45, 100, 115])

MAX_DIST = 350
MINE_DELAY = 7.5
BLACKLIST_TIME = 25.0

# **НОВЕ**: Налаштування Текстурного Снайпера
# Будь-який камінь має контрастну текстуру. Гладкі тіні або дороги ігноруються.
# Якщо бот перестав бити камені, знизь це число до 7.0. Якщо б'є тіні - підвищ до 9.0.
TEXTURE_THRESHOLD = 8.3

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
template_names = ['ok', 'error', 'wait', 'server_list', 'server_2', 'server_3', 'server_4', 'back', 'play_now']

for name in template_names:
    path = f'ui/{name}.png'
    if os.path.exists(path):
        UI_TEMPLATES[name] = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    else:
        UI_TEMPLATES[name] = None


def check_ui_element(gray_img, template, threshold=0.92):
    """Розумний зір для табличок і меню (з масштабуванням)"""
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


def check_strict_element(gray_img, template, threshold=0.96):
    """Снайперський зір + Мікро-масштабування (пробиває стиснення нижніх вікон)"""
    if template is None: return None
    scales = [1.0, 0.98, 0.96, 0.94, 0.92, 0.90, 0.88, 1.02, 1.04]
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

    play_pos = check_ui_element(gray_img, UI_TEMPLATES.get('play_now'), threshold=0.88)
    if play_pos:
        return {"action": "click", "x": play_pos[0], "y": play_pos[1],
                "msg": "Викинуло в головне меню! Тисну Play Now."}

    back_pos = check_ui_element(gray_img, UI_TEMPLATES.get('back'), threshold=0.90)
    if back_pos:
        return {"action": "click", "x": back_pos[0], "y": back_pos[1],
                "msg": "Зайшов не туди (приватний сервер)! Тисну Back."}

    current_threshold = 0.91 if target_server == "server_4" else 0.96

    server_pos = check_strict_element(gray_img, UI_TEMPLATES.get(target_server), threshold=current_threshold)
    if server_pos:
        return {"action": "click", "x": server_pos[0], "y": server_pos[1],
                "msg": f"Знайшов {target_server}! Заходжу на сервер."}

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

                # **НОВЕ**: Текстурний Снайпер
                # Беремо тільки сіру зону об'єкта
                local_mask = np.zeros((bh, bw), dtype=np.uint8)
                cv2.drawContours(local_mask, [cnt - [bx, by]], -1, 255, -1)
                roi_gray = gray[by:by + bh, bx:bx + bw]

                # Рахуємо контрастність текстури
                pixels = roi_gray[local_mask == 255]
                if len(pixels) > 0:
                    std_dev = np.std(pixels)

                    # Ігноруємо гладкі об'єкти (тіні)
                    if std_dev < TEXTURE_THRESHOLD: continue
                else:
                    continue

                dist = np.sqrt((cx - center_x) ** 2 + (cy - center_y) ** 2)
                score = dist + (np.mean(pixels) * 2)
                if dist <= MAX_DIST:
                    targets.append((cx, cy, score))

            if not targets:
                return {"type": "empty"}

            targets.sort(key=lambda x: x[2])
            return {"type": "mine", "cx": targets[0][0], "cy": targets[0][1]}

    except Exception as e:
        print(f"Vision error: {e}")
        return None


print("=== Бот запущено: Версія 'Текстурний Снайпер' ===\n")
time.sleep(3)

try:
    while True:
        now = time.time()

        for w in WINDOWS:
            w["blacklist"] = [item for item in w["blacklist"] if item[2] > now]

            result = find_target(w, now)

            if result:
                if result["type"] == "empty":
                    w["empty_scans"] += 1
                    if w["empty_scans"] == 15:
                        print(f"[{w['name']}] Каменів немає. Пересуваю камеру...")
                        cx = int((w["roi"][0] + (w["roi"][2] - w["roi"][0]) // 2) * SX)
                        cy = int((w["roi"][1] + (w["roi"][3] - w["roi"][1]) // 2) * SY)

                        dx = random.choice([-250, 250])
                        dy = random.choice([-250, 250])

                        pyautogui.moveTo(cx, cy, duration=0.2)
                        pyautogui.dragTo(cx + dx, cy + dy, duration=0.5, button='left')

                        w["empty_scans"] = 0
                        w["blacklist"].clear()
                        w["next_mine"] = now + 1.5
                else:
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
                        time.sleep(0.05)
                        pyautogui.click()
                        w["next_ui"] = now + 2.5

                    elif result["action"] == "scroll":
                        pyautogui.moveTo(fx, fy - 15, duration=0.2)
                        pyautogui.click()
                        time.sleep(0.1)

                        pyautogui.moveTo(fx, fy + 150, duration=0.2)
                        time.sleep(0.1)

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

        time.sleep(0.02)

except KeyboardInterrupt:
    print("\n[СТОП] Бот зупинений.")