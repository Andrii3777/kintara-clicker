# -*- coding: utf-8 -*-
"""
reconnect.py
============
Слой ПЕРЕЗАХОДА на сервер. ОБЩИЙ для всех ботов (как win_input).

Логика взята из стороннего скрипта: ловим UI-попапы дисконнекта/очереди/ошибки
и меню выбора сервера через matchTemplate, возвращаем действие. Сам клик/скролл
делает бот (через win_input). Тут — ТОЛЬКО зрение, без захвата экрана и мыши.

Поток (приоритет сверху вниз, emergency_ui_check):
  wait        -> в очереди, сервер стартует -> ждать, ничего не жать
  ok          -> попап вылета -> клик OK
  error/RETRY -> ошибка коннекта -> клик RETRY
  server_N    -> нужный сервер виден -> клик, заходим
  server_list -> список открыт, сервера не видно -> скролл вниз искать

Шаблоны сняты на ЧУЖОМ разрешении -> matchTemplate мульти-скейлом терпит
небольшую разницу, но при сильном расхождении ПЕРЕСНЯТЬ под свой экран.
Координаты в действиях — ЛОКАЛЬНЫЕ внутри региона (бот добавит offset региона).
"""

import os
import cv2


TEMPLATE_NAMES = ['ok', 'error', 'wait', 'server_list',
                  'server_2', 'server_3', 'server_4']

# Масштабы для подгонки под чужое разрешение (matchTemplate не масштаб-инвариантен)
SCALES = [1.0, 0.95, 0.90, 0.85, 0.80, 1.05, 1.10]


def load_templates(ui_dir):
    """Загрузить PNG-шаблоны из ui_dir в gray. Нет файла -> None под именем."""
    templates = {}
    for name in TEMPLATE_NAMES:
        path = os.path.join(ui_dir, f"{name}.png")
        templates[name] = (cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                           if os.path.exists(path) else None)
    return templates


def check_ui_element(gray_img, template, threshold=0.92):
    """Мульти-скейл matchTemplate. Нашёл -> (cx,cy) центра совпадения, иначе None."""
    if template is None:
        return None
    for scale in SCALES:
        w = int(template.shape[1] * scale)
        h = int(template.shape[0] * scale)
        if (w < 10 or h < 10
                or w > gray_img.shape[1] or h > gray_img.shape[0]):
            continue
        resized = cv2.resize(template, (w, h))
        res = cv2.matchTemplate(gray_img, resized, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        if max_val >= threshold:
            return (max_loc[0] + w // 2, max_loc[1] + h // 2)
    return None


def emergency_ui_check(gray_img, target_server, templates):
    """
    Проверить кадр (gray) на UI перезахода. Вернуть dict действия или None.
    target_server: 'server_2'|'server_3'|'server_4' — какой сервер этого аккаунта.

    Действия:
      {'action':'wait'}                      — ждать (ничего не жать)
      {'action':'click','x':..,'y':..}       — клик в (x,y) локально
      {'action':'scroll','x':..,'y':..}      — мотать список от (x,y) заголовка
    """
    if check_ui_element(gray_img, templates.get('wait'), threshold=0.95):
        return {"action": "wait", "msg": "Очередь/запуск сервера — жду."}

    ok_pos = check_ui_element(gray_img, templates.get('ok'), threshold=0.90)
    if ok_pos:
        return {"action": "click", "x": ok_pos[0], "y": ok_pos[1],
                "msg": "Вылет — жму OK."}

    error_pos = check_ui_element(gray_img, templates.get('error'), threshold=0.90)
    if error_pos:
        return {"action": "click", "x": error_pos[0], "y": error_pos[1],
                "msg": "Ошибка коннекта — жму RETRY."}

    # строгий порог: не путать заголовок с кнопками
    srv_pos = check_ui_element(gray_img, templates.get(target_server),
                               threshold=0.93)
    if srv_pos:
        return {"action": "click", "x": srv_pos[0], "y": srv_pos[1],
                "msg": f"Нашёл {target_server} — захожу."}

    list_pos = check_ui_element(gray_img, templates.get('server_list'),
                                threshold=0.90)
    if list_pos:
        return {"action": "scroll", "x": list_pos[0], "y": list_pos[1],
                "msg": f"{target_server} не виден — мотаю список."}

    return None
