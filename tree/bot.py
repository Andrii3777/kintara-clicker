# -*- coding: utf-8 -*-
"""
bot.py
======
Авто-кликер по деревьям. Захват экрана -> детекция -> клик -> умная пауза.

Логика паузы (нюансы из обсуждения):
  - Первое дерево: last_pos пуст -> ждём фикс FIRST_WAIT (перс может быть далеко).
  - Дальше: перс стоит у прошлого дерева (last_pos). Берём ближайшее новое
    дерево. Пауза = CHOP_TIME (руб) + дистанция_px * SEC_PER_PX (дойти) + джиттер.
    Близко = ждём мало, далеко = ждём больше.

Отбор деревьев:
  - Кликаем только УВЕРЕННЫЕ (крупный ствол + крупная крона) — ствол хорошо виден.
    Слабые пропускаем, деревья респавнятся, не жалко.

Стоп: горячая клавиша 'q'.
Безопасность: DRY_RUN=True печатает действие БЕЗ клика. pyautogui FAILSAFE вкл —
  резко увести мышь в угол экрана = аварийный стоп.

Запуск:
  python bot.py            # боевой режим (кликает)
  правь DRY_RUN ниже для теста без кликов
"""

import time
import math
import random

import cv2
import numpy as np

# DPI-aware ПЕРВЫМ делом — до mss/любого захвата. Иначе mss выставит
# system-DPI по первому монитору и клики по второму монитору уедут.
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "common"))   # общий win_input
import win_input
win_input.set_dpi_aware()

import keyboard

from tree_detector import detect_trees, highlight_count
from reconnect import load_templates, emergency_ui_check
from pan import Panner


# ============================================================
# CONFIG — крутим тут
# ============================================================

DRY_RUN = False            # True = печать без клика (СНАЧАЛА проверь так!)

# Какой монитор сканировать. mss: [1]=первый, [2]=второй, [0]=все вместе.
# Узнать свои: python -c "import mss;[print(i,m) for i,m in enumerate(mss.mss().monitors)]"
MONITOR_INDEX = 1

# --- Ожидание срубки по ПОДСВЕТКЕ (у ДЕРЕВА, камера статична) ---
# Глоу появляется когда перс ДОШЁЛ до дерева и рубит (не при клике!),
# горит весь руб, гаснет когда срублено. Камера СТАТИЧНА -> дерево на
# фикс-позиции экрана = точка клика. Следим за глоу ТАМ:
#   глоу появился -> перс дошёл, рубит (бюджет ожидания учитывает ходьбу)
#   глоу погас    -> срублено -> дальше
CHOP_POLL = 0.2           # как часто проверять экран, сек
TREE_GLOW_R = 40          # радиус зоны у дерева (точка клика) где ищем глоу, px
# Глоу ловим ОТНОСИТЕЛЬНО фона (base), а не абсолютом. В густом лесу зона может
# и без рубки иметь яркие пиксели -> абсолют врал (всё уходило в timeout).
# base = глоу в точке В МОМЕНТ клика (перс ещё не дошёл, рубки нет).
NEON_RISE_PX = 180        # подъём над base -> перс дошёл, РУБИТ
NEON_FALL_PX = 60         # вернулось к base+это -> СРУБЛЕНО (глоу погас)
# Появления глоу ждём с учётом ходьбы: чем дальше дерево, тем дольше идти.
APPEAR_BASE = 3.0         # база ожидания появления глоу, сек
WALK_SEC_PER_PX = 0.020   # + сек на 1px дистанции (дойти) — щедро, не бросать цель
APPEAR_MAX = 6.0          # потолок ожидания появления, сек (остаток до HARD_CAP — на руб)
VANISH_CONFIRM = 2        # глоу погас N раз подряд -> точно срублено
HARD_CAP = 8.0            # ЖЁСТКИЙ потолок: что бы ни было -> дальше, сек
MAX_CHOP_WAIT = 30.0      # (оставлен; реально ограничивает HARD_CAP)
POST_CHOP_PAUSE = (0.1, 0.3)  # пауза после срубки (имитация человека), сек
DRY_WAIT = 2.0            # в DRY_RUN срубку не видно -> фикс пауза, сек

NO_TREE_WAIT = 1.0        # пауза после пана до пересканирования, сек

# --- Панорамирование карты (камера статична, карта = РОМБ/изометрия!) ---
# Логика обхода — общая в common/pan.py (Panner). Тут только включатель + параметры.
PAN_ENABLED = False       # ВЫКЛ по умолчанию: целей нет -> просто ждать на месте,
                          # карту НЕ двигать. True -> обход «газонокосилкой».
# Оси карты на экране идут по ДИАГОНАЛЯМ (↗↙ и ↘↖), не горизонт/вертикаль.
# Поэтому панорамируем вдоль диагоналей: ряд вдоль одной оси, шаг вдоль другой.
# Всё по счётчикам (детект края убран — у края игра пружинит). Заезд в пустоту
# безвреден: деревьев там нет, едем дальше.
PAN_DIST = 380            # длина drag по каждой оси, px (диагональ ~PAN_DIST*1.4)
PAN_DURATION = 0.6        # время одного drag, сек
PAN_SETTLE = 0.6          # пауза после drag чтоб карта устаканилась, сек
# Размах должен покрывать ВЕСЬ ромб угол-в-угол. Мало -> видит только полосу
# у старта (лево+верх). Правый/нижний угол достаёт ось ШАГА (SE-марш): он несёт
# вид в bottom-right. Не доходит до права/низа -> увеличь PAN_NUM_ROWS (число
# SE-шагов). Старт у top-left края: первые шаги пружинят (счётчик тикает, карта
# стоит) -> закладываем запас.
PAN_ROW_LEN = 12          # панов вдоль ряда (ось SW<->NE: низ-лево + верх-право)
PAN_NUM_ROWS = 20         # шагов вдоль оси SE<->NW (несёт в право-низ), с запасом
# Потолок «карта обойдена» Panner считает сам из PAN_ROW_LEN/PAN_NUM_ROWS.
PAN_ROW_START = 'SW'      # ось ряда: 'SW'(↙) <-> 'NE'(↗)
PAN_STEP_START = 'SE'     # ось шага: 'SE'(↘) <-> 'NW'(↖)

# --- Отбор деревьев (уверенность) ---
MIN_CLICK_TRUNK_AREA = 12   # ствол меньше -> пропуск
MIN_CLICK_CROWN_AREA = 80   # крона меньше -> пропуск

# --- Выбор цели ---
# Якорь персонажа для ПЕРВОГО выбора (доли экрана). Перс ~ центр.
CHAR_ANCHOR = (0.42, 0.40)
SAME_TREE_R = 18          # радиус "это ТО ЖЕ дерево" между кадрами (джиттер кроны).
                          # Дедуп срубленного/чёрного списка. МАЛЕНЬКИЙ: больше ->
                          # съедает соседние деревья и бот идёт к дальнему.
COOLDOWN_SEC = 12.0       # не кликать дерево возле недавней точки клика
                          # (страховка от зацикливания если не деспавнилось)
BLACKLIST_TTL = 45.0      # дерево дало no_highlight (куст/недоступно) -> в чёрный
                          # список на столько сек, жёстко не кликаем

# --- Прочее ---
HOTKEY_STOP = 'q'
# (pyautogui-failsafe убран — клик теперь через win_input. Стоп только по 'q'.)

# --- Перезаход на сервер (UI-шаблоны matchTemplate, common/reconnect.py) ---
# Высший приоритет: видим попап вылета/очереди/меню сервера -> бросаем фарм,
# жмём OK/RETRY/SERVER N или мотаем список, потом фарм дальше. Шаблоны в common/ui/.
UI_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "..", "common", "ui")
TARGET_SERVER = "server_2"   # на какой сервер заходит этот аккаунт
UI_CHECK_INTERVAL = 1.0      # как часто сканить UI когда спокойно, сек
UI_CLICK_DELAY = 2.5         # пауза после клика по UI, сек
UI_WAIT_DELAY = 4.0          # пауза если «в очереди», сек
UI_SCROLL_DELAY = 1.8        # пауза после шага скролла списка, сек
UI_SCROLL_FOCUS_DY = 15      # клик на столько px ВЫШЕ заголовка (фокус окна)
UI_SCROLL_INTO_DY = 150      # курсор на столько px НИЖЕ заголовка (в зону списка)
UI_SCROLL_NOTCHES = -1       # шаг колеса: <0 вниз (1 щелчок)


# ============================================================
# Утилиты
# ============================================================

def grab_screen():
    """Захват выбранного монитора -> BGR для OpenCV (через mss, быстро)."""
    import mss
    with mss.mss() as sct:
        mon = sct.monitors[MONITOR_INDEX]
        shot = np.array(sct.grab(mon))
        return cv2.cvtColor(shot, cv2.COLOR_BGRA2BGR), mon


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def pick_confident(trees):
    """Оставить только уверенные деревья (хорошо виден ствол+крона)."""
    return [t for t in trees
            if t['trunk_area'] >= MIN_CLICK_TRUNK_AREA
            and t['crown_area'] >= MIN_CLICK_CROWN_AREA]


def choose_target(trees, last_pos, anchor_px, recent, blacklist):
    """
    Выбрать дерево для клика.
      - last_pos None  -> ближайшее к якорю персонажа.
      - last_pos задан -> ПРОСТО ближайшее к прошлому дереву (без отсечки по
        дистанции — иначе сосед впритык выпадает и бот идёт к дальнему).
        Срубленное дерево уже исключено через recent (дедуп SAME_TREE_R).
      - recent: недавние клики — МЯГКИЙ кулдаун (можно вернуться если ничего нет).
      - blacklist: дерево дало no_highlight (куст/декор/недоступно) — ЖЁСТКО
        исключаем, без фолбэка. Если исключили всё -> None -> пан.
    Возвращает дерево или None.
    """
    ref = last_pos if last_pos is not None else anchor_px

    def near(p, pts):
        return any(dist(p, r) < SAME_TREE_R for r in pts)

    # жёсткий фильтр: чёрный список (ложные/недоступные) — никогда не берём
    usable = [t for t in trees if not near(t['click'], blacklist)]

    # мягкий фильтр: кулдаун недавних (вкл. срубленное). БЕЗ отсечки по дистанции
    # от last_pos — иначе сосед впритык выпадает и бот идёт к дальнему.
    cand = [t for t in usable if not near(t['click'], recent)]
    if not cand:                     # всё на мягком кулдауне -> вернёмся к usable
        cand = usable
    if not cand:                     # всё в чёрном списке -> цели нет
        return None

    return min(cand, key=lambda t: dist(t['click'], ref))


def wait_until_chopped(tree_px, dist_px):
    """
    Ждём пока перс не срубит дерево — по ПОДСВЕТКЕ в точке дерева (камера статична).
    Возвращает (status, stats): status = 'chopped'|'no_highlight'|'timeout'|'stop',
    stats = (base, peak, last) — пиксели глоу для лога/калибровки.

    Глоу меряем ОТНОСИТЕЛЬНО base (фон в момент клика, рубки ещё нет):
      present = base + NEON_RISE_PX  -> перс дошёл, рубит
      gone    = base + NEON_FALL_PX  -> срублено
    Так густой лес (яркий фон) не путает абсолютный порог.

    Фаза 1: ждём появления глоу (перс дошёл). Бюджет = база + ходьба.
    Фаза 2: ждём пока глоу ПОГАСНЕТ (срублено).
    """
    tx, ty = tree_px
    start = time.time()
    deadline = start + HARD_CAP                       # жёсткий потолок на всё
    appear_deadline = min(start + APPEAR_BASE + dist_px * WALK_SEC_PER_PX,
                          start + APPEAR_MAX, deadline)

    bgr, _ = grab_screen()
    base = highlight_count(bgr, tx, ty, TREE_GLOW_R)  # фон до прихода перса
    present_thr = base + NEON_RISE_PX
    gone_thr = base + NEON_FALL_PX
    peak = base
    last = base

    # --- фаза 1: дождаться появления глоу у дерева ---
    seen = False
    while time.time() < appear_deadline:
        if keyboard.is_pressed(HOTKEY_STOP):
            return 'stop', (base, peak, last)
        bgr, _ = grab_screen()
        last = highlight_count(bgr, tx, ty, TREE_GLOW_R)
        peak = max(peak, last)
        if last >= present_thr:
            seen = True
            break
        time.sleep(CHOP_POLL)
    if not seen:
        # не дождались глоу: клик мимо / не дошёл -> дальше
        status = 'no_highlight' if time.time() < deadline else 'timeout'
        return status, (base, peak, last)

    # --- фаза 2: дождаться пока глоу погаснет ---
    gone = 0
    while True:
        if keyboard.is_pressed(HOTKEY_STOP):
            return 'stop', (base, peak, last)
        if time.time() > deadline:                    # жёсткий потолок -> дальше
            return 'timeout', (base, peak, last)

        bgr, _ = grab_screen()
        last = highlight_count(bgr, tx, ty, TREE_GLOW_R)
        peak = max(peak, last)
        if last <= gone_thr:
            gone += 1                      # глоу пропал
            if gone >= VANISH_CONFIRM:     # стабильно N раз -> срублено
                return 'chopped', (base, peak, last)
        else:
            gone = 0                       # глоу ещё горит -> рубится

        time.sleep(CHOP_POLL)


# ============================================================
# Перезаход на сервер
# ============================================================

def check_reconnect(bgr, mon, templates, state):
    """
    UI-перезаход (высший приоритет). Кадр -> OCR -> действие.
    Вернуть задержку next_ui (сек) если было действие, иначе None (UI нет).
    Координаты действия локальные -> добавляем смещение монитора.
    state — dict памяти выбора сервера (скролл/детект низа).
    """
    # OCR-перезаход кормим ЦВЕТНЫМ кадром (в gray теряется текст кнопок).
    action = emergency_ui_check(bgr, TARGET_SERVER, templates, state)
    if action is None:
        return None
    print(f"  [UI] {action['msg']}")
    act = action['action']
    if act == 'wait':
        return UI_WAIT_DELAY
    ax = action['x'] + mon['left']
    ay = action['y'] + mon['top']
    if act == 'click':
        if not DRY_RUN:
            win_input.click_abs(ax, ay)
        return UI_CLICK_DELAY
    if act == 'scroll':
        # Фокус окна на БЕЗОПАСНОЙ точке (заголовок, не карточка!) -> колесо над
        # списком. reconnect отдаёт focus_x/focus_y (заголовок) и x/y (зона колеса).
        if not DRY_RUN:
            fx = action.get('focus_x', action['x']) + mon['left']
            fy = action.get('focus_y', action['y'] - UI_SCROLL_FOCUS_DY) + mon['top']
            win_input.click_abs(fx, fy)
            time.sleep(0.1)
            win_input.scroll(UI_SCROLL_NOTCHES, ax, ay)
        return UI_SCROLL_DELAY
    return None


# ============================================================
# Главный цикл
# ============================================================

def main():
    print("=" * 50)
    print(" TREE BOT")
    print(f"  DRY_RUN = {DRY_RUN}  (True = без кликов)")
    print(f"  Монитор = [{MONITOR_INDEX}]  DPI-режим = {win_input.set_dpi_aware()}")
    print(f"  Виртуальный стол = {win_input.virtual_screen()}")
    print(f"  Стоп: клавиша '{HOTKEY_STOP}'")
    print("=" * 50)
    print("Старт через 3 сек — переключись в игру...")
    time.sleep(3)

    templates = load_templates(UI_DIR)
    if templates.get('_ocr') is None:
        print("  ВНИМАНИЕ: OCR недоступен — перезаход выключен (фарм работает).")
    next_ui = 0.0      # троттл UI-проверки перезахода
    block_until = 0.0  # НЕ фармить пока идёт UI-действие (очередь/клик/скролл)
    ui_state = {}      # память выбора сервера (скролл по списку, детект низа)

    last_pos = None
    clicks = 0
    recent = []        # [(pos, время_клика)] — недавние клики для кулдауна
    blacklist = []     # [(pos, время)] — ложные/недоступные (no_highlight)
    # пан-обход карты — общий Panner (common/pan.py). Создаётся лениво на первый
    # пан (нужен mon из захвата). Вся логика змейки внутри.
    pan_cfg = {'PAN_DIST': PAN_DIST, 'PAN_DURATION': PAN_DURATION,
               'PAN_SETTLE': PAN_SETTLE, 'PAN_ROW_LEN': PAN_ROW_LEN,
               'PAN_NUM_ROWS': PAN_NUM_ROWS, 'ROW_START': PAN_ROW_START,
               'STEP_START': PAN_STEP_START, 'DRY_WAIT': NO_TREE_WAIT}
    panner = None

    while True:
        if keyboard.is_pressed(HOTKEY_STOP):
            print("Стоп по клавише.")
            break

        # чистим память по времени
        now = time.time()
        recent = [(p, t) for (p, t) in recent if now - t < COOLDOWN_SEC]
        recent_pts = [p for (p, _) in recent]
        blacklist = [(p, t) for (p, t) in blacklist if now - t < BLACKLIST_TTL]
        blacklist_pts = [p for (p, _) in blacklist]

        bgr, mon = grab_screen()

        # перезаход на сервер — высший приоритет (свой троттл next_ui)
        if now >= next_ui:
            delay = check_reconnect(bgr, mon, templates, ui_state)
            if delay is not None:
                next_ui = now + delay
                block_until = now + delay   # пока UI активен — НЕ фармить
                last_pos = None      # после перезахода координаты другие
                recent = []
                if panner:
                    panner.reset()   # карта/координаты другие
                continue
            next_ui = now + UI_CHECK_INTERVAL

        # UI-действие в процессе (очередь/перезаход) — деревья не трогаем
        if now < block_until:
            time.sleep(0.05)
            continue

        H, W = bgr.shape[:2]
        anchor_px = (int(CHAR_ANCHOR[0] * W), int(CHAR_ANCHOR[1] * H))

        trees, _, _, _ = detect_trees(bgr)
        trees = pick_confident(trees)

        target = None
        if trees:
            target = choose_target(trees, last_pos, anchor_px,
                                   recent_pts, blacklist_pts)

        # нет деревьев ИЛИ все в чёрном списке (ложные кусты)
        if target is None:
            reason = "деревьев нет" if not trees else "все цели в чёрном списке"
            if not PAN_ENABLED:
                time.sleep(NO_TREE_WAIT)   # пан выключен -> ждём на месте
                continue
            if panner is None:
                panner = Panner(mon, pan_cfg, dry_run=DRY_RUN, tag=str(clicks),
                                grab=lambda: grab_screen()[0])
            if panner.step(reason, before=bgr):
                break                      # карта обойдена -> стоп
            last_pos = None                # карта сдвинулась
            recent = []
            continue

        if panner:
            panner.reset()   # есть реальная цель -> сбрасываем счётчик панов

        cx, cy = target['click']
        # mss отдаёт координаты внутри монитора; добавляем смещение монитора
        screen_x = cx + mon['left']
        screen_y = cy + mon['top']

        d = 0.0 if last_pos is None else dist(last_pos, target['click'])
        print(f"[{clicks}] дерево@({cx},{cy}) "
              f"trunk={target['trunk_area']} crown={target['crown_area']} "
              f"dist={d:.0f}px -> click({screen_x},{screen_y})")

        if not DRY_RUN:
            win_input.click_abs(screen_x, screen_y)

        last_pos = target['click']
        recent.append((target['click'], time.time()))
        clicks += 1

        # --- ждём ПОКА не срубим (по факту исчезновения дерева) ---
        if DRY_RUN:
            # клика не было -> дерево не исчезнет, ждём фикс чтоб логика крутилась
            time.sleep(DRY_WAIT)
        else:
            t0 = time.time()
            res, (gbase, gpeak, glast) = wait_until_chopped(target['click'], d)
            took = time.time() - t0
            if res == 'stop':
                print("Стоп по клавише.")
                break
            print(f"    -> {res} за {took:.1f}s "
                  f"(глоу base={gbase} peak={gpeak} last={glast})")
            if res == 'no_highlight':
                # глоу не появился -> куст/декор/недоступно -> в чёрный список,
                # больше не кликаем это место (иначе вечный цикл по одной точке)
                blacklist.append((target['click'], time.time()))
                print(f"    -> в чёрный список {target['click']}")
                continue
            # человеческая пауза после срубки
            time.sleep(random.uniform(*POST_CHOP_PAUSE))

    print(f"Готово. Кликов: {clicks}")


if __name__ == "__main__":
    main()
