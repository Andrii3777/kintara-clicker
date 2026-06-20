# -*- coding: utf-8 -*-
"""
bot.py (КАМЕНЬ)
===============
Авто-кликер по камням. Захват экрана -> детекция -> клик -> ждать добычу -> пан.
Скелет тот же что у дерева (tree/bot.py): механика одинаковая, отличается
детектор (stone_detector) и сигнатура цели.

Логика ожидания добычи — по ПОДСВЕТКЕ (глоу) в точке камня, камера статична:
  глоу появился -> перс дошёл и добывает (бюджет учитывает ходьбу)
  глоу погас    -> добыто -> следующий камень
ВНИМАНИЕ: глоу добычи камня ПОКА не замерен (highlight_count = копия из дерева).
Переснять в живой игре (см. diag_glow-аналог / лог 'глоу base/peak/last').

Отбор камней:
  Кликаем только УВЕРЕННЫЕ (площадь >= MIN_CLICK_STONE_AREA). Мелкие крошки/шум
  пропускаем — камни респавнятся, не жалко.

Стоп: горячая клавиша 'q'. DRY_RUN=True печатает действие БЕЗ клика.

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

from stone_detector import detect_stones, stone_px
from reconnect import load_templates, emergency_ui_check


# ============================================================
# CONFIG — крутим тут
# ============================================================

DRY_RUN = False            # True = печать без клика (СНАЧАЛА проверь так!)

# Какой монитор сканировать. mss: [1]=первый, [2]=второй, [0]=все вместе.
# Узнать свои: python -c "import mss;[print(i,m) for i,m in enumerate(mss.mss().monitors)]"
MONITOR_INDEX = 1

# --- Ожидание добычи по ПИКСЕЛЯМ КАМНЯ (камера статична) ---
# У камня НЕТ зелёного глоу (замер diag_capture). Сигнал ИНВЕРТИРОВАН против
# дерева: перс встаёт на камень и заслоняет его телом -> пиксели камня в зоне
# ПАДАЮТ (добывает); отошёл -> ВЕРНУЛИСЬ (добыто). base = пиксели камня в момент
# клика (камень виден, перс ещё идёт).
#   px <= base*ENGAGE_DROP_FRAC -> перс пришёл, ДОБЫВАЕТ (заслонил)
#   px >= base*RECOVER_FRAC      -> ДОБЫТО (перс отошёл, камень снова виден)
CHOP_POLL = 0.2           # как часто проверять экран, сек
STONE_GLOW_R = 40         # радиус зоны у камня (точка клика), px
ENGAGE_DROP_FRAC = 0.70   # px упал ниже этой доли base -> перс взялся/добывает
RECOVER_FRAC = 0.90       # px вернулся выше этой доли base -> добыто
# Прихода перса ждём с учётом ходьбы: чем дальше камень, тем дольше идти.
APPEAR_BASE = 3.0         # база ожидания "перс взялся", сек
WALK_SEC_PER_PX = 0.020   # + сек на 1px дистанции (дойти) — щедро, не бросать цель
APPEAR_MAX = 6.0          # потолок ожидания "перс взялся", сек (остаток до cap — на добычу)
VANISH_CONFIRM = 2        # px вернулся N раз подряд -> точно добыто
HARD_CAP = 8.0            # ЖЁСТКИЙ потолок: максимум между добычами, сек
POST_CHOP_PAUSE = (0.1, 0.3)  # пауза после добычи (имитация человека), сек
DRY_WAIT = 2.0            # в DRY_RUN добычу не видно -> фикс пауза, сек

NO_TARGET_WAIT = 1.0      # пауза после пана до пересканирования, сек

# --- Панорамирование карты (камера статична, карта = РОМБ/изометрия!) ---
# Оси карты на экране идут по ДИАГОНАЛЯМ (↗↙ и ↘↖). Панорамируем вдоль диагоналей:
# ряд вдоль одной оси, шаг вдоль другой. Всё по счётчикам (детект края убран —
# у края игра пружинит). Те же значения что у дерева (карта та же).
PAN_DIST = 380            # длина drag по каждой оси, px
PAN_DURATION = 0.6        # время одного drag, сек
PAN_SETTLE = 0.6          # пауза после drag чтоб карта устаканилась, сек
PAN_ROW_LEN = 12          # панов вдоль ряда (ось SW<->NE)
PAN_NUM_ROWS = 20         # шагов вдоль оси SE<->NW (несёт в право-низ), с запасом
MAX_EMPTY_PANS = 2 * PAN_NUM_ROWS * (PAN_ROW_LEN + 1) + 8  # стоп: обошли всё
PAN_ROW_START = 'SW'      # ось ряда: 'SW'(↙) <-> 'NE'(↗)
PAN_STEP_START = 'SE'     # ось шага: 'SE'(↘) <-> 'NW'(↖)

# --- Отбор камней (уверенность) ---
# Детектор уже фильтрует площадь/форму/окружение. Тут — доп. порог "уверенного"
# камня для клика (крупный явный камень). Мелкие далёкие пропускаем.
MIN_CLICK_STONE_AREA = 200

# --- Выбор цели ---
# Якорь персонажа для ПЕРВОГО выбора (доли экрана). Перс ~ центр.
CHAR_ANCHOR = (0.42, 0.40)
SAME_STONE_R = 18         # радиус "это ТОТ ЖЕ камень" между кадрами (джиттер).
                          # Дедуп добытого/чёрного списка. МАЛЕНЬКИЙ: больше ->
                          # съедает соседний камень и бот идёт к дальнему.
COOLDOWN_SEC = 12.0       # не кликать камень возле недавней точки (антизацикл)
BLACKLIST_TTL = 45.0      # камень дал no_highlight (ложный/строение/недоступно)
                          # -> в чёрный список на столько сек, жёстко не кликаем

# --- Прочее ---
HOTKEY_STOP = 'q'

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


def frame_change(a, b):
    """Доля заметно изменившихся пикселей между кадрами a и b (0..1)."""
    diff = cv2.absdiff(a, b)
    return float(np.count_nonzero(diff.sum(2) > 30)) / (a.shape[0] * a.shape[1])


# Диагональные направления (вдоль осей ромба) в экранных (dx, dy).
DIR_VEC = {'NE': (1, -1), 'SW': (-1, 1), 'NW': (-1, -1), 'SE': (1, 1)}
DIR_FLIP = {'NE': 'SW', 'SW': 'NE', 'NW': 'SE', 'SE': 'NW'}


def pan_and_measure(bgr_before, mon, direction):
    """Перетянуть карту по диагонали direction и вернуть долю изменений (лог)."""
    if DRY_RUN:
        time.sleep(NO_TARGET_WAIT)
        return 1.0
    W, H = mon['width'], mon['height']
    sx = mon['left'] + W // 2
    sy = mon['top'] + H // 2
    vx, vy = DIR_VEC[direction]
    win_input.drag_abs(sx, sy, sx + vx * PAN_DIST, sy + vy * PAN_DIST,
                       steps=25, duration=PAN_DURATION)
    time.sleep(PAN_SETTLE)
    after, _ = grab_screen()
    return frame_change(bgr_before, after)


def pick_confident(stones):
    """Оставить только уверенные камни (крупная площадь)."""
    return [s for s in stones if s['area'] >= MIN_CLICK_STONE_AREA]


def choose_target(stones, last_pos, anchor_px, recent, blacklist):
    """
    Выбрать камень для клика.
      - last_pos None  -> ближайший к якорю персонажа.
      - last_pos задан -> ПРОСТО ближайший к прошлому камню (без отсечки по
        дистанции — иначе сосед впритык выпадает и бот идёт к дальнему).
        Добытый камень уже исключён через recent (дедуп SAME_STONE_R).
      - recent: недавние клики — МЯГКИЙ кулдаун (можно вернуться если ничего нет).
      - blacklist: камень дал no_highlight (ложный/строение/недоступно) — ЖЁСТКО
        исключаем. Если исключили всё -> None -> пан.
    Возвращает камень или None.
    """
    ref = last_pos if last_pos is not None else anchor_px

    def near(p, pts):
        return any(dist(p, r) < SAME_STONE_R for r in pts)

    usable = [s for s in stones if not near(s['click'], blacklist)]
    cand = [s for s in usable if not near(s['click'], recent)]
    if not cand:
        cand = usable
    if not cand:
        return None
    return min(cand, key=lambda s: dist(s['click'], ref))


def wait_until_mined(stone_pos, dist_px):
    """
    Ждём пока перс не добудет камень — по ПИКСЕЛЯМ КАМНЯ в точке (камера статична).
    Возвращает (status, stats): status = 'mined'|'no_highlight'|'timeout'|'stop',
    stats = (base, lowest, last) — пиксели камня для лога/калибровки.

    Сигнал ИНВЕРТИРОВАН (у камня нет глоу): base = пиксели камня в момент клика.
      px <= base*ENGAGE_DROP_FRAC -> перс пришёл, заслонил, ДОБЫВАЕТ
      px >= base*RECOVER_FRAC      -> ДОБЫТО (перс отошёл)
    Фаза 1: ждём падения (перс взялся). Не упало за бюджет -> цель недостижима/
            ложная (строение) -> no_highlight -> чёрный список.
    Фаза 2: ждём восстановления (добыто).
    """
    tx, ty = stone_pos
    start = time.time()
    deadline = start + HARD_CAP
    appear_deadline = min(start + APPEAR_BASE + dist_px * WALK_SEC_PER_PX,
                          start + APPEAR_MAX, deadline)

    bgr, _ = grab_screen()
    base = max(1, stone_px(bgr, tx, ty, STONE_GLOW_R))  # камень виден, перс идёт
    drop_thr = base * ENGAGE_DROP_FRAC
    recover_thr = base * RECOVER_FRAC
    lowest = base
    last = base

    # --- фаза 1: дождаться падения (перс пришёл, заслонил/добывает) ---
    engaged = False
    while time.time() < appear_deadline:
        if keyboard.is_pressed(HOTKEY_STOP):
            return 'stop', (base, lowest, last)
        bgr, _ = grab_screen()
        last = stone_px(bgr, tx, ty, STONE_GLOW_R)
        lowest = min(lowest, last)
        if last <= drop_thr:
            engaged = True
            break
        time.sleep(CHOP_POLL)
    if not engaged:
        # перс не взялся: клик мимо / строение / недостижимо -> чёрный список
        status = 'no_highlight' if time.time() < deadline else 'timeout'
        return status, (base, lowest, last)

    # --- фаза 2: дождаться восстановления (перс отошёл = добыто) ---
    back = 0
    while True:
        if keyboard.is_pressed(HOTKEY_STOP):
            return 'stop', (base, lowest, last)
        if time.time() > deadline:
            return 'timeout', (base, lowest, last)

        bgr, _ = grab_screen()
        last = stone_px(bgr, tx, ty, STONE_GLOW_R)
        lowest = min(lowest, last)
        if last >= recover_thr:
            back += 1
            if back >= VANISH_CONFIRM:
                return 'mined', (base, lowest, last)
        else:
            back = 0

        time.sleep(CHOP_POLL)


# ============================================================
# Перезаход на сервер
# ============================================================

def check_reconnect(bgr, mon, templates):
    """
    UI-перезаход (высший приоритет). Кадр -> matchTemplate -> действие.
    Вернуть задержку next_ui (сек) если было действие, иначе None (UI нет).
    Координаты шаблона локальные -> добавляем смещение монитора.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    action = emergency_ui_check(gray, TARGET_SERVER, templates)
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
        # клик чуть ВЫШЕ заголовка (фокус) -> курсор в зону списка -> колесо
        if not DRY_RUN:
            win_input.click_abs(ax, ay - UI_SCROLL_FOCUS_DY)
            time.sleep(0.1)
            win_input.scroll(UI_SCROLL_NOTCHES, ax, ay + UI_SCROLL_INTO_DY)
        return UI_SCROLL_DELAY
    return None


# ============================================================
# Главный цикл
# ============================================================

def main():
    print("=" * 50)
    print(" STONE BOT")
    print(f"  DRY_RUN = {DRY_RUN}  (True = без кликов)")
    print(f"  Монитор = [{MONITOR_INDEX}]  DPI-режим = {win_input.set_dpi_aware()}")
    print(f"  Виртуальный стол = {win_input.virtual_screen()}")
    print(f"  Стоп: клавиша '{HOTKEY_STOP}'")
    print("=" * 50)
    print("Старт через 3 сек — переключись в игру...")
    time.sleep(3)

    templates = load_templates(UI_DIR)
    missing = [n for n, t in templates.items() if t is None]
    if missing:
        print(f"  ВНИМАНИЕ: нет UI-шаблонов {missing} — перезаход для них не сработает.")
    next_ui = 0.0      # троттл UI-проверки перезахода

    last_pos = None
    clicks = 0
    recent = []        # [(pos, время)] — недавние клики для кулдауна
    blacklist = []     # [(pos, время)] — ложные/недоступные (no_highlight)
    empty_pans = 0
    row_dir = PAN_ROW_START
    step_dir = PAN_STEP_START
    row_pans = 0
    step_count = 0
    pending_step = False

    def do_pan(bgr, mon, reason):
        """Пан-обход. Возвращает True если пора остановиться (обошли всё)."""
        nonlocal empty_pans, last_pos, recent
        nonlocal row_dir, step_dir, row_pans, step_count, pending_step
        empty_pans += 1
        if empty_pans > MAX_EMPTY_PANS:
            print(f"[{clicks}] {MAX_EMPTY_PANS} панов без целей — стоп (обошли карту).")
            return True

        if pending_step:
            direction = step_dir
            changed = pan_and_measure(bgr, mon, direction)
            step_count += 1
            print(f"[{clicks}] {reason} -> шаг '{direction}' "
                  f"измен={changed*100:.0f}% ряд {step_count}/{PAN_NUM_ROWS} "
                  f"({empty_pans}/{MAX_EMPTY_PANS})")
            if step_count >= PAN_NUM_ROWS:
                step_count = 0
                step_dir = DIR_FLIP[step_dir]
                print(f"    -> все ряды пройдены: разворот шага '{step_dir}'")
            pending_step = False
        else:
            direction = row_dir
            changed = pan_and_measure(bgr, mon, direction)
            row_pans += 1
            print(f"[{clicks}] {reason} -> пан '{direction}' "
                  f"измен={changed*100:.0f}% ряд {row_pans}/{PAN_ROW_LEN} "
                  f"({empty_pans}/{MAX_EMPTY_PANS})")
            if row_pans >= PAN_ROW_LEN:
                row_pans = 0
                pending_step = True
                row_dir = DIR_FLIP[row_dir]
                print(f"    -> ряд пройден: шаг '{step_dir}', обратно '{row_dir}'")

        last_pos = None
        recent = []
        return False

    while True:
        if keyboard.is_pressed(HOTKEY_STOP):
            print("Стоп по клавише.")
            break

        now = time.time()
        recent = [(p, t) for (p, t) in recent if now - t < COOLDOWN_SEC]
        recent_pts = [p for (p, _) in recent]
        blacklist = [(p, t) for (p, t) in blacklist if now - t < BLACKLIST_TTL]
        blacklist_pts = [p for (p, _) in blacklist]

        bgr, mon = grab_screen()

        # перезаход на сервер — высший приоритет (свой троттл next_ui)
        if now >= next_ui:
            delay = check_reconnect(bgr, mon, templates)
            if delay is not None:
                next_ui = now + delay
                last_pos = None      # после перезахода координаты другие
                recent = []
                continue
            next_ui = now + UI_CHECK_INTERVAL

        H, W = bgr.shape[:2]
        anchor_px = (int(CHAR_ANCHOR[0] * W), int(CHAR_ANCHOR[1] * H))

        stones, _, _, _ = detect_stones(bgr)
        stones = pick_confident(stones)

        target = None
        if stones:
            target = choose_target(stones, last_pos, anchor_px,
                                   recent_pts, blacklist_pts)

        # нет камней ИЛИ все в чёрном списке -> пан
        if target is None:
            reason = "камней нет" if not stones else "все цели в чёрном списке"
            if do_pan(bgr, mon, reason):
                break
            continue

        empty_pans = 0   # есть реальная цель -> сброс счётчика панов

        cx, cy = target['click']
        screen_x = cx + mon['left']
        screen_y = cy + mon['top']

        d = 0.0 if last_pos is None else dist(last_pos, target['click'])
        print(f"[{clicks}] камень@({cx},{cy}) area={target['area']} "
              f"dist={d:.0f}px -> click({screen_x},{screen_y})")

        if not DRY_RUN:
            win_input.click_abs(screen_x, screen_y)

        last_pos = target['click']
        recent.append((target['click'], time.time()))
        clicks += 1

        if DRY_RUN:
            time.sleep(DRY_WAIT)
        else:
            t0 = time.time()
            res, (gbase, glow, glast) = wait_until_mined(target['click'], d)
            took = time.time() - t0
            if res == 'stop':
                print("Стоп по клавише.")
                break
            print(f"    -> {res} за {took:.1f}s "
                  f"(камень_px base={gbase} min={glow} last={glast})")
            if res == 'no_highlight':
                # глоу не появился -> ложный/строение/недоступно -> в чёрный список
                blacklist.append((target['click'], time.time()))
                print(f"    -> в чёрный список {target['click']}")
                continue
            time.sleep(random.uniform(*POST_CHOP_PAUSE))

    print(f"Готово. Кликов: {clicks}")


if __name__ == "__main__":
    main()
