# -*- coding: utf-8 -*-
"""
stone-coal-4x/bot.py
====================
Авто-кликер камня+угля на 4 АККАУНТА одновременно. Экран = 4 окна игры,
разложенные «плиткой» 2x2 в квадрантах выбранного монитора (каждый квадрант =
свой логин/мир).

ПОЧЕМУ ТАК (ключевой принцип):
  Курсор в системе ОДИН (win_input = SendInput, абсолютные координаты
  виртуального стола). 4 бота не могут двигать его одновременно. НО почти всё
  время бот ЖДЁТ (перс идёт + добывает, до HARD_CAP сек). Пока акк-0 ждёт, курсор
  свободен -> кликаем акк-1, акк-2, акк-3. Получаем ~4x за счёт ИНТЕРЛИВИНГА
  пауз, а не за счёт второго курсора.

Как реализовано:
  - Каждый аккаунт = конечный автомат (FSM): SCAN -> WAIT_ENGAGE -> WAIT_RECOVER
    -> (mined) -> SCAN, либо -> PAN если целей нет.
  - Планировщик по кругу даёт каждому аккаунту ОДИН «тик»: максимум один захват
    региона + максимум одно действие мышью. Блокирующего ожидания нет.
  - Все дедлайны по time.time() -> интерливинг их не ломает.

ОТЛИЧИЕ ОТ ДЕРЕВА: у камня НЕТ зелёного глоу. Сигнал добычи ИНВЕРТИРОВАН: перс
встаёт на камень, заслоняет телом -> пиксели камня в зоне ПАДАЮТ (добывает);
отошёл -> ВЕРНУЛИСЬ (добыто). base = пиксели камня в момент клика.
  px <= base*ENGAGE_DROP_FRAC -> перс пришёл, ДОБЫВАЕТ
  px >= base*RECOVER_FRAC      -> ДОБЫТО

Регионы: авто-2x2 квадранты MONITOR_INDEX (см. build_regions).
Зависимости (НЕ дублируем): stone_detector из ../stone-coal, win_input из ../common.

Стоп: 'q'. DRY_RUN=True — без кликов (FSM крутится на фикс-паузах).
Запуск (из папки stone-coal-4x):  python bot.py
"""

import time
import math
import random

import cv2
import numpy as np

# DPI-aware ПЕРВЫМ делом — до mss/любого захвата.
# Бутстрап путей: ../common (win_input, общий) и ../stone-coal (stone_detector).
# Детектор НЕ дублируем — единственный источник в stone-coal/.
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "common"))
sys.path.insert(0, os.path.join(_HERE, "..", "stone-coal"))
import win_input
win_input.set_dpi_aware()

import keyboard

from stone_detector import detect_stones, stone_px
from reconnect import load_templates, emergency_ui_check


# ============================================================
# CONFIG — крутим тут (значения = как в одиночном stone-coal/bot.py)
# ============================================================

DRY_RUN = False            # True = печать без клика (СНАЧАЛА проверь так!)
MONITOR_INDEX = 1          # какой монитор делить на 2x2

# --- Раскладка квадрантов ---
REGION_INSET = 0.0         # 0..0.2; >0 если по краям квадранта мусор/чужое окно

# Какие квадранты активны (аккаунты). Выключи лишние -> можно гонять 2 или 3 акка.
# TL=лево-верх, TR=право-верх, BL=лево-низ, BR=право-низ.
QUADRANT_TL = True
QUADRANT_TR = True
QUADRANT_BL = True
QUADRANT_BR = True

# --- Перезаход на сервер (UI-шаблоны matchTemplate) ---
# Каждый квадрант (аккаунт) заходит на СВОЙ сервер. Шаблоны в common/ui/.
# Высший приоритет: видим попап дисконнекта/очереди/меню -> бросаем фарм,
# обрабатываем (клик OK/RETRY/SERVER N или скролл списка), потом фарм дальше.
UI_DIR = os.path.join(_HERE, "..", "common", "ui")
QUADRANT_TL_SERVER = "server_2"   # лево-верх
QUADRANT_TR_SERVER = "server_3"   # право-верх
QUADRANT_BL_SERVER = "server_4"   # лево-низ
QUADRANT_BR_SERVER = "server_2"   # право-низ (поправь под нужный сервер)
UI_CHECK_INTERVAL = 1.0    # как часто сканить UI когда спокойно, сек
UI_CLICK_DELAY = 2.5       # пауза после клика по UI, сек
UI_WAIT_DELAY = 4.0        # пауза если «в очереди», сек
UI_SCROLL_DELAY = 1.8      # пауза после шага скролла списка, сек
UI_SCROLL_FOCUS_DY = 15    # клик на столько px ВЫШЕ заголовка (фокус окна)
UI_SCROLL_INTO_DY = 150    # курсор на столько px НИЖЕ заголовка (в зону списка)
UI_SCROLL_NOTCHES = -1     # шаг колеса: <0 вниз (1 щелчок)

# --- Ожидание добычи по ПИКСЕЛЯМ КАМНЯ (инвертированный сигнал, без глоу) ---
CHOP_POLL = 0.2            # целевой интервал между поллами ОДНОГО аккаунта, сек
STONE_GLOW_R = 40
ENGAGE_DROP_FRAC = 0.70    # px <= base*это -> перс взялся, добывает (заслонил)
RECOVER_FRAC = 0.90        # px >= base*это -> добыто (перс отошёл)
APPEAR_BASE = 3.0
WALK_SEC_PER_PX = 0.020
APPEAR_MAX = 6.0
VANISH_CONFIRM = 2
HARD_CAP = 8.0
POST_CHOP_PAUSE = (0.1, 0.3)
DRY_WAIT = 2.0             # в DRY_RUN держим «добычу» столько

# --- Панорамирование (по диагоналям ромба, как в одиночном боте) ---
PAN_DIST = 380
PAN_DURATION = 0.6
PAN_SETTLE = 0.6
PAN_ROW_LEN = 12
PAN_NUM_ROWS = 20
MAX_EMPTY_PANS = 2 * PAN_NUM_ROWS * (PAN_ROW_LEN + 1) + 8
PAN_ROW_START = 'SW'
PAN_STEP_START = 'SE'

# --- Отбор/выбор цели ---
# ВНИМАНИЕ: квадрант = 1/4 экрана -> камни мельче. Порог площади, возможно,
# надо уменьшить против одиночного бота. Откалибруй на debug-кадре квадранта.
MIN_CLICK_STONE_AREA = 200
CHAR_ANCHOR = (0.42, 0.40)   # доли ВНУТРИ квадранта (перс ~ центр окна)
SAME_STONE_R = 18
COOLDOWN_SEC = 12.0
BLACKLIST_TTL = 45.0

HOTKEY_STOP = 'q'

DIR_VEC = {'NE': (1, -1), 'SW': (-1, 1), 'NW': (-1, -1), 'SE': (1, 1)}
DIR_FLIP = {'NE': 'SW', 'SW': 'NE', 'NW': 'SE', 'SE': 'NW'}


# ============================================================
# Утилиты
# ============================================================

def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def frame_change(a, b):
    diff = cv2.absdiff(a, b)
    return float(np.count_nonzero(diff.sum(2) > 30)) / (a.shape[0] * a.shape[1])


def build_regions():
    """
    Разбить MONITOR_INDEX на 4 квадранта 2x2 и оставить только АКТИВНЫЕ
    (QUADRANT_*). Каждый регион — dict в АБСОЛЮТНЫХ координатах виртуального стола
    (left/top уже со смещением монитора).
    """
    import mss
    with mss.mss() as sct:
        mon = sct.monitors[MONITOR_INDEX]
    L, T, W, H = mon['left'], mon['top'], mon['width'], mon['height']
    hw, hh = W // 2, H // 2
    ins_x, ins_y = int(hw * REGION_INSET), int(hh * REGION_INSET)
    cells = {'TL': (L, T), 'TR': (L + hw, T),
             'BL': (L, T + hh), 'BR': (L + hw, T + hh)}
    enabled = {'TL': QUADRANT_TL, 'TR': QUADRANT_TR,
               'BL': QUADRANT_BL, 'BR': QUADRANT_BR}
    servers = {'TL': QUADRANT_TL_SERVER, 'TR': QUADRANT_TR_SERVER,
               'BL': QUADRANT_BL_SERVER, 'BR': QUADRANT_BR_SERVER}
    regions = []
    for nm in ('TL', 'TR', 'BL', 'BR'):   # стабильный порядок
        if not enabled[nm]:
            continue
        cl, ct = cells[nm]
        regions.append({'name': nm,
                        'left': cl + ins_x, 'top': ct + ins_y,
                        'width': hw - 2 * ins_x, 'height': hh - 2 * ins_y,
                        'target_server': servers[nm]})
    if not regions:
        raise SystemExit("Все квадранты выключены — включи хотя бы один QUADRANT_*.")
    return regions


def grab_region(region):
    """Захват одного региона -> BGR. Координаты региона уже абсолютные."""
    import mss
    with mss.mss() as sct:
        shot = np.array(sct.grab({'left': region['left'], 'top': region['top'],
                                  'width': region['width'],
                                  'height': region['height']}))
    return cv2.cvtColor(shot, cv2.COLOR_BGRA2BGR)


def pick_confident(stones):
    return [s for s in stones if s['area'] >= MIN_CLICK_STONE_AREA]


def choose_target(stones, last_pos, anchor_px, recent, blacklist):
    """Та же логика выбора, что в одиночном боте (см. stone-coal/bot.py)."""
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


# ============================================================
# Аккаунт = конечный автомат (без блокирующих ожиданий)
# ============================================================

# Состояния FSM
SCAN = 'SCAN'
WAIT_ENGAGE = 'WAIT_ENGAGE'    # ждём падения px (перс заслонил, добывает)
WAIT_RECOVER = 'WAIT_RECOVER'  # ждём восстановления px (перс отошёл = добыто)
DONE = 'DONE'


class AccountBot:
    """
    Один аккаунт в своём квадранте. tick() двигает автомат НА ОДИН шаг и делает
    максимум одно действие мышью. Возвращает True если этот аккаунт закончил
    (обошёл всю карту и целей нет).
    """

    def __init__(self, idx, region, templates):
        self.idx = idx
        self.region = region
        self.tag = f"A{idx}:{region['name']}"
        self.state = SCAN

        # перезаход на сервер
        self.templates = templates
        self.target_server = region['target_server']
        self.next_ui = 0.0      # троттл UI-проверки

        # выбор цели
        self.last_pos = None
        self.recent = []        # [(pos, t)]
        self.blacklist = []     # [(pos, t)]
        self.clicks = 0

        # текущая цель / тайминги ожидания
        self.target = None
        self.t0 = 0.0
        self.appear_deadline = 0.0
        self.hard_deadline = 0.0
        self.base = 1
        self.drop_thr = 0
        self.recover_thr = 0
        self.lowest = 0
        self.last = 0
        self.back = 0
        self.resume_at = 0.0
        self.last_poll = 0.0

        # пан-обход (счётчики, как в одиночном do_pan)
        self.empty_pans = 0
        self.row_dir = PAN_ROW_START
        self.step_dir = PAN_STEP_START
        self.row_pans = 0
        self.step_count = 0
        self.pending_step = False

    # -- помощники --
    def _abs(self, local_xy):
        return (self.region['left'] + local_xy[0],
                self.region['top'] + local_xy[1])

    def _stone_px(self, bgr):
        cx, cy = self.target['click']
        return stone_px(bgr, cx, cy, STONE_GLOW_R)

    def _expire_memory(self, now):
        self.recent = [(p, t) for (p, t) in self.recent if now - t < COOLDOWN_SEC]
        self.blacklist = [(p, t) for (p, t) in self.blacklist
                          if now - t < BLACKLIST_TTL]

    def _reset_farm(self):
        """Сброс фарм-состояния (после перезахода карта/координаты другие)."""
        self.state = SCAN
        self.target = None
        self.last_pos = None
        self.recent = []
        self.resume_at = 0.0

    def _tick_ui(self, now):
        """
        UI/перезаход — высший приоритет. Вернуть True если было UI-действие
        (фарм в этот тик пропускаем). False -> UI нет, фармим дальше.
        """
        bgr = grab_region(self.region)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        action = emergency_ui_check(gray, self.target_server, self.templates)
        if action is None:
            self.next_ui = now + UI_CHECK_INTERVAL
            return False

        print(f"[{self.tag}] {action['msg']}")
        self._reset_farm()
        act = action['action']

        if act == 'wait':
            self.next_ui = now + UI_WAIT_DELAY
            return True

        ax, ay = self._abs((action['x'], action['y']))
        if act == 'click':
            if not DRY_RUN:
                win_input.click_abs(ax, ay)
            self.next_ui = now + UI_CLICK_DELAY
        elif act == 'scroll':
            # клик чуть ВЫШЕ заголовка (фокус) -> курсор в зону списка -> колесо
            if not DRY_RUN:
                win_input.click_abs(ax, ay - UI_SCROLL_FOCUS_DY)
                time.sleep(0.1)
                win_input.scroll(UI_SCROLL_NOTCHES, ax, ay + UI_SCROLL_INTO_DY)
            self.next_ui = now + UI_SCROLL_DELAY
        return True

    # -- главный тик --
    def tick(self):
        now = time.time()
        if self.state == DONE:
            return True

        # перезаход на сервер — высший приоритет, свой троттл next_ui
        if now >= self.next_ui:
            if self._tick_ui(now):
                return False

        # троттл: один аккаунт не чаще CHOP_POLL (другие тикают свободно)
        if now - self.last_poll < CHOP_POLL:
            return False
        self.last_poll = now

        if self.state == SCAN:
            return self._tick_scan(now)
        if self.state == WAIT_ENGAGE:
            self._tick_engage(now)
            return False
        if self.state == WAIT_RECOVER:
            self._tick_recover(now)
            return False
        return False

    def _tick_scan(self, now):
        if now < self.resume_at:
            return False
        self._expire_memory(now)
        bgr = grab_region(self.region)
        H, W = bgr.shape[:2]
        anchor = (int(CHAR_ANCHOR[0] * W), int(CHAR_ANCHOR[1] * H))

        stones, _, _, _ = detect_stones(bgr)
        stones = pick_confident(stones)

        recent_pts = [p for (p, _) in self.recent]
        black_pts = [p for (p, _) in self.blacklist]
        target = choose_target(stones, self.last_pos, anchor,
                               recent_pts, black_pts) if stones else None

        if target is None:
            reason = "нет камней" if not stones else "все в чёрном списке"
            return self._do_pan(bgr, reason)

        self.empty_pans = 0
        self.target = target
        d = 0.0 if self.last_pos is None else dist(self.last_pos, target['click'])

        ax, ay = self._abs(target['click'])
        print(f"[{self.tag} #{self.clicks}] камень@{target['click']} "
              f"area={target['area']} dist={d:.0f} -> click({ax},{ay})")

        if not DRY_RUN:
            win_input.click_abs(ax, ay)

        self.last_pos = target['click']
        self.recent.append((target['click'], now))
        self.clicks += 1

        if DRY_RUN:
            self.resume_at = now + DRY_WAIT
            self.state = SCAN
            return False

        # старт ожидания добычи
        self.t0 = now
        self.hard_deadline = now + HARD_CAP
        self.appear_deadline = min(now + APPEAR_BASE + d * WALK_SEC_PER_PX,
                                   now + APPEAR_MAX, self.hard_deadline)
        self.base = max(1, self._stone_px(bgr))  # камень виден, перс ещё идёт
        self.drop_thr = self.base * ENGAGE_DROP_FRAC
        self.recover_thr = self.base * RECOVER_FRAC
        self.lowest = self.base
        self.last = self.base
        self.back = 0
        self.state = WAIT_ENGAGE
        return False

    def _tick_engage(self, now):
        bgr = grab_region(self.region)
        self.last = self._stone_px(bgr)
        self.lowest = min(self.lowest, self.last)
        if self.last <= self.drop_thr:
            self.back = 0
            self.state = WAIT_RECOVER
            return
        if now >= self.appear_deadline:
            # перс не взялся -> ложный/строение/недоступно -> чёрный список
            self.blacklist.append((self.target['click'], now))
            print(f"[{self.tag} #{self.clicks}] no_highlight "
                  f"(base={self.base} min={self.lowest}) -> чёрный список "
                  f"{self.target['click']}")
            self.state = SCAN

    def _tick_recover(self, now):
        if now >= self.hard_deadline:
            print(f"[{self.tag} #{self.clicks}] timeout "
                  f"(base={self.base} min={self.lowest} last={self.last})")
            self.resume_at = now + random.uniform(*POST_CHOP_PAUSE)
            self.state = SCAN
            return
        bgr = grab_region(self.region)
        self.last = self._stone_px(bgr)
        self.lowest = min(self.lowest, self.last)
        if self.last >= self.recover_thr:
            self.back += 1
            if self.back >= VANISH_CONFIRM:
                took = now - self.t0
                print(f"[{self.tag} #{self.clicks}] mined за {took:.1f}s "
                      f"(base={self.base} min={self.lowest} last={self.last})")
                self.resume_at = now + random.uniform(*POST_CHOP_PAUSE)
                self.state = SCAN
        else:
            self.back = 0

    def _do_pan(self, bgr_before, reason):
        """Один шаг пан-обхода (короткий блокирующий drag в свой тик).
        Возвращает True если аккаунт обошёл всю карту -> DONE."""
        self.empty_pans += 1
        if self.empty_pans > MAX_EMPTY_PANS:
            print(f"[{self.tag}] {MAX_EMPTY_PANS} панов без целей — стоп (карта обойдена).")
            self.state = DONE
            return True

        if self.pending_step:
            direction = self.step_dir
            changed = self._pan(bgr_before, direction)
            self.step_count += 1
            print(f"[{self.tag}] {reason} -> шаг '{direction}' "
                  f"измен={changed*100:.0f}% ряд {self.step_count}/{PAN_NUM_ROWS} "
                  f"({self.empty_pans}/{MAX_EMPTY_PANS})")
            if self.step_count >= PAN_NUM_ROWS:
                self.step_count = 0
                self.step_dir = DIR_FLIP[self.step_dir]
            self.pending_step = False
        else:
            direction = self.row_dir
            changed = self._pan(bgr_before, direction)
            self.row_pans += 1
            print(f"[{self.tag}] {reason} -> пан '{direction}' "
                  f"измен={changed*100:.0f}% ряд {self.row_pans}/{PAN_ROW_LEN} "
                  f"({self.empty_pans}/{MAX_EMPTY_PANS})")
            if self.row_pans >= PAN_ROW_LEN:
                self.row_pans = 0
                self.pending_step = True
                self.row_dir = DIR_FLIP[self.row_dir]

        self.last_pos = None       # карта сдвинулась
        self.recent = []
        self.state = SCAN
        return False

    def _pan(self, bgr_before, direction):
        """Drag из ЦЕНТРА своего квадранта по диагонали. Не выходит за регион."""
        if DRY_RUN:
            time.sleep(0.1)
            return 1.0
        r = self.region
        sx = r['left'] + r['width'] // 2
        sy = r['top'] + r['height'] // 2
        vx, vy = DIR_VEC[direction]
        win_input.drag_abs(sx, sy, sx + vx * PAN_DIST, sy + vy * PAN_DIST,
                           steps=25, duration=PAN_DURATION)
        time.sleep(PAN_SETTLE)
        after = grab_region(r)
        return frame_change(bgr_before, after)


# ============================================================
# Планировщик: по кругу тикает 4 аккаунта, владеет единым курсором
# ============================================================

def main():
    regions = build_regions()
    print("=" * 56)
    print(" STONE-COAL BOT x4 (interleave, единый курсор)")
    print(f"  DRY_RUN = {DRY_RUN}")
    print(f"  Монитор = [{MONITOR_INDEX}]  DPI = {win_input.set_dpi_aware()}")
    print(f"  Виртуальный стол = {win_input.virtual_screen()}")
    for r in regions:
        print(f"    {r['name']}: left={r['left']} top={r['top']} "
              f"{r['width']}x{r['height']}  сервер={r['target_server']}")
    templates = load_templates(UI_DIR)
    missing = [n for n, t in templates.items() if t is None]
    print(f"  UI-шаблоны: {UI_DIR}")
    if missing:
        print(f"  ВНИМАНИЕ: нет шаблонов {missing} — перезаход для них не сработает.")
    print(f"  Стоп: '{HOTKEY_STOP}'")
    print("=" * 56)
    print("Старт через 3 сек — разложи 4 окна по квадрантам...")
    time.sleep(3)

    bots = [AccountBot(i, r, templates) for i, r in enumerate(regions)]

    while True:
        if keyboard.is_pressed(HOTKEY_STOP):
            print("Стоп по клавише.")
            break
        all_done = True
        for b in bots:
            if keyboard.is_pressed(HOTKEY_STOP):
                break
            done = b.tick()
            all_done = all_done and done
        if all_done:
            print("Все аккаунты обошли карты — стоп.")
            break
        time.sleep(0.01)   # лёгкий yield (троттл — внутри tick)

    total = sum(b.clicks for b in bots)
    print(f"Готово. Кликов всего: {total} "
          f"({', '.join(f'{b.tag}={b.clicks}' for b in bots)})")


if __name__ == "__main__":
    main()
