# -*- coding: utf-8 -*-
"""
tree-4x/bot.py
==============
Авто-кликер деревьев на 4 АККАУНТА одновременно. Экран = 4 окна игры,
разложенные «плиткой» 2x2 в квадрантах выбранного монитора (каждый квадрант =
свой логин/мир).

ПОЧЕМУ ТАК (ключевой принцип):
  Курсор в системе ОДИН (win_input = SendInput, абсолютные координаты
  виртуального стола). 4 бота не могут двигать его одновременно. НО почти всё
  время бот ЖДЁТ (перс идёт + рубит, до HARD_CAP сек). Пока акк-0 ждёт глоу,
  курсор свободен -> кликаем акк-1, акк-2, акк-3. Получаем ~4x за счёт
  ИНТЕРЛИВИНГА пауз, а не за счёт второго курсора.

Как реализовано:
  - Каждый аккаунт = конечный автомат (FSM): SCAN -> WAIT_APPEAR -> WAIT_VANISH
    -> (chopped) -> SCAN, либо -> PAN если целей нет.
  - Планировщик (scheduler) по кругу даёт каждому аккаунту ОДИН «тик»: максимум
    один захват региона + максимум одно действие мышью (клик ИЛИ один шаг пана).
    Блокирующего ожидания нет -> аккаунты не держат друг друга.
  - Все дедлайны по time.time() (не по итерациям) -> интерливинг их не ломает.

Регионы: авто-2x2 квадранты MONITOR_INDEX (см. build_regions).

Детекция/глоу — те же tree_detector.detect_trees / highlight_count, что и в
одиночном bot.py. Картинка просто меньше (квадрант) -> возможно подкрутить
пороги площади (см. MIN_CLICK_*).

Стоп: 'q'. DRY_RUN=True — без кликов (FSM крутится на фикс-паузах).

Зависимости (НЕ дублируем): tree_detector из ../tree, win_input из ../common.
Запуск (из папки tree-4x):  python bot.py
"""

import time
import math
import random

import cv2
import numpy as np

# DPI-aware ПЕРВЫМ делом — до mss/любого захвата.
# Бутстрап путей: ../common (win_input, общий) и ../tree (tree_detector, ядро CV).
# Детектор НЕ дублируем — единственный источник в tree/.
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "common"))
sys.path.insert(0, os.path.join(_HERE, "..", "tree"))
import win_input
win_input.set_dpi_aware()

import keyboard

from tree_detector import detect_trees, highlight_count
from reconnect import load_templates, emergency_ui_check
from pan import Panner, zoom_map


# ============================================================
# CONFIG — крутим тут (значения = как в одиночном bot.py)
# ============================================================

DRY_RUN = False            # True = печать без клика (СНАЧАЛА проверь так!)
MONITOR_INDEX = 1          # какой монитор делить на 2x2

# --- Раскладка квадрантов ---
# Авто 2x2 монитора. Поджимаем рамки внутрь на REGION_INSET долю, чтобы НЕ
# ловить чужой квадрант / границу окна по краю.
REGION_INSET = 0.0         # 0..0.2; >0 если по краям квадранта мусор

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

# --- Ожидание срубки по ГЛОУ (относительно base) ---
CHOP_POLL = 0.2            # целевой интервал между поллами ОДНОГО аккаунта, сек
TREE_GLOW_R = 40
NEON_RISE_PX = 180         # глоу >= base+это -> перс дошёл, рубит
NEON_FALL_PX = 60          # глоу <= base+это -> срублено
APPEAR_BASE = 3.0
WALK_SEC_PER_PX = 0.020
APPEAR_MAX = 6.0
VANISH_CONFIRM = 2
HARD_CAP = 8.0
POST_CHOP_PAUSE = (0.1, 0.3)
DRY_WAIT = 2.0             # в DRY_RUN держим «рубку» столько

# --- Панорамирование (логика общая в common/pan.py, Panner) ---
PAN_ENABLED = False        # ВЫКЛ по умолчанию: целей нет -> аккаунт ждёт на месте,
                           # карту НЕ двигает. True -> обход «газонокосилкой».
PAN_DIST = 380
PAN_DURATION = 0.6
PAN_SETTLE = 0.6
PAN_ROW_LEN = 12
PAN_NUM_ROWS = 20
PAN_ROW_START = 'SW'
PAN_STEP_START = 'SE'
PAN_DRY_WAIT = 0.1         # пауза вместо drag в DRY_RUN

# --- Zoom карты после входа на сервер (общий, common/pan.py) ---
ZOOM_ON_ENTER = True       # после захода уменьшить карту колесом
ZOOM_NOTCHES = -1          # <0 вниз (обычно zoom out; наоборот -> >0)
ZOOM_TIMES = 6             # сколько полных прокрутов

# --- Отбор/выбор цели ---
# ВНИМАНИЕ: квадрант = 1/4 экрана -> деревья мельче. Пороги площади, возможно,
# надо уменьшить против одиночного bot.py. Откалибруй на debug-кадре квадранта.
MIN_CLICK_TRUNK_AREA = 12
MIN_CLICK_CROWN_AREA = 80
CHAR_ANCHOR = (0.42, 0.40)   # доли ВНУТРИ квадранта (перс ~ центр окна)
SAME_TREE_R = 18
COOLDOWN_SEC = 12.0
BLACKLIST_TTL = 45.0

HOTKEY_STOP = 'q'

# общий cfg пана для Panner (одинаков для всех квадрантов)
PAN_CFG = {'PAN_DIST': PAN_DIST, 'PAN_DURATION': PAN_DURATION,
           'PAN_SETTLE': PAN_SETTLE, 'PAN_ROW_LEN': PAN_ROW_LEN,
           'PAN_NUM_ROWS': PAN_NUM_ROWS, 'ROW_START': PAN_ROW_START,
           'STEP_START': PAN_STEP_START, 'DRY_WAIT': PAN_DRY_WAIT}


# ============================================================
# Утилиты
# ============================================================

def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def build_regions():
    """
    Разбить MONITOR_INDEX на 4 квадранта 2x2 и оставить только АКТИВНЫЕ
    (QUADRANT_*). Каждый регион — dict в АБСОЛЮТНЫХ координатах виртуального стола
    (left/top уже со смещением монитора), как ждёт mss.grab и win_input.
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


def pick_confident(trees):
    return [t for t in trees
            if t['trunk_area'] >= MIN_CLICK_TRUNK_AREA
            and t['crown_area'] >= MIN_CLICK_CROWN_AREA]


def choose_target(trees, last_pos, anchor_px, recent, blacklist):
    """Та же логика выбора, что в одиночном боте (см. bot.py.choose_target)."""
    ref = last_pos if last_pos is not None else anchor_px

    def near(p, pts):
        return any(dist(p, r) < SAME_TREE_R for r in pts)

    usable = [t for t in trees if not near(t['click'], blacklist)]
    cand = [t for t in usable if not near(t['click'], recent)]
    if not cand:
        cand = usable
    if not cand:
        return None
    return min(cand, key=lambda t: dist(t['click'], ref))


# ============================================================
# Аккаунт = конечный автомат (без блокирующих ожиданий)
# ============================================================

# Состояния FSM
SCAN = 'SCAN'
WAIT_APPEAR = 'WAIT_APPEAR'
WAIT_VANISH = 'WAIT_VANISH'
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
        self.block_until = 0.0  # НЕ фармить пока идёт UI-действие (очередь/клик)
        self.ui_state = {}      # память выбора сервера (скролл/детект низа)

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
        self.base = 0
        self.peak = 0
        self.last = 0
        self.gone = 0
        self.resume_at = 0.0     # не сканировать раньше (пауза после срубки)
        self.last_poll = 0.0     # время последнего полла (троттл CHOP_POLL)

        # пан-обход — общий Panner (drag из центра СВОЕГО квадранта)
        self.panner = Panner(region, PAN_CFG, dry_run=DRY_RUN, tag=self.tag,
                             grab=lambda: grab_region(self.region))
        self.was_in_ui = True   # True на старте -> при первом входе в мир зумим

    # -- помощники --
    def _abs(self, local_xy):
        """Локальная точка региона -> абс. координаты виртуального стола."""
        return (self.region['left'] + local_xy[0],
                self.region['top'] + local_xy[1])

    def _glow(self, bgr):
        cx, cy = self.target['click']
        return highlight_count(bgr, cx, cy, TREE_GLOW_R)

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
        self.panner.reset()   # карта сдвинулась — снейк с начала

    def _tick_ui(self, now):
        """
        UI/перезаход — высший приоритет. Вернуть True если было UI-действие
        (фарм в этот тик пропускаем). False -> UI нет, фармим дальше.
        """
        bgr = grab_region(self.region)
        # OCR-перезаход кормим ЦВЕТНЫМ кадром (в gray теряется текст кнопок).
        action = emergency_ui_check(bgr, self.target_server, self.templates,
                                    self.ui_state)
        if action is None:
            self.next_ui = now + UI_CHECK_INTERVAL
            # вошли в мир после UI/перезахода -> уменьшить карту (один раз)
            if self.was_in_ui:
                self.was_in_ui = False
                if ZOOM_ON_ENTER:
                    zoom_map(self.region, ZOOM_NOTCHES, ZOOM_TIMES,
                             dry_run=DRY_RUN, log=lambda m: print(f"[{self.tag}]{m}"))
                    self._reset_farm()
                    return True   # тик потрачен на zoom
            return False

        print(f"[{self.tag}] {action['msg']}")
        self._reset_farm()      # НЕ трогает ui_state (память скролла ведёт reconnect)
        self.was_in_ui = True   # были в UI -> при выходе в мир зумнём
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
            # Фокус-клик ТОЛЬКО если reconnect дал безопасную точку (заголовок).
            # Нет focus_x -> заголовок не распознан -> НЕ кликаем (иначе попадём
            # по карточке платного KINTARA CLUB), крутим только колесом.
            if not DRY_RUN:
                if 'focus_x' in action:
                    fx, fy = self._abs((action['focus_x'], action['focus_y']))
                    win_input.click_abs(fx, fy)
                    time.sleep(0.1)
                win_input.scroll(UI_SCROLL_NOTCHES, ax, ay)
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
                self.block_until = self.next_ui   # пока UI активен — НЕ фармить
                return False

        # UI-действие в процессе (очередь/перезаход) — фарм не трогаем
        if now < self.block_until:
            return False

        # троттл: один аккаунт не чаще CHOP_POLL (но другие тикают свободно)
        if now - self.last_poll < CHOP_POLL:
            return False
        self.last_poll = now

        if self.state == SCAN:
            return self._tick_scan(now)
        if self.state == WAIT_APPEAR:
            self._tick_appear(now)
            return False
        if self.state == WAIT_VANISH:
            self._tick_vanish(now)
            return False
        return False

    def _tick_scan(self, now):
        if now < self.resume_at:
            return False
        self._expire_memory(now)
        bgr = grab_region(self.region)
        H, W = bgr.shape[:2]
        anchor = (int(CHAR_ANCHOR[0] * W), int(CHAR_ANCHOR[1] * H))

        trees, _, _, _ = detect_trees(bgr)
        trees = pick_confident(trees)

        recent_pts = [p for (p, _) in self.recent]
        black_pts = [p for (p, _) in self.blacklist]
        target = choose_target(trees, self.last_pos, anchor,
                               recent_pts, black_pts) if trees else None

        if target is None:
            reason = "нет деревьев" if not trees else "все в чёрном списке"
            if not PAN_ENABLED:
                self.resume_at = now + 1.0   # пан выкл -> ждём на месте
                return False
            if self.panner.step(reason, before=bgr):
                self.state = DONE            # карта обойдена
                return True
            self.last_pos = None             # карта сдвинулась
            self.recent = []
            self.state = SCAN
            return False

        self.panner.reset()
        self.target = target
        d = 0.0 if self.last_pos is None else dist(self.last_pos, target['click'])

        ax, ay = self._abs(target['click'])
        print(f"[{self.tag} #{self.clicks}] дерево@{target['click']} "
              f"trunk={target['trunk_area']} crown={target['crown_area']} "
              f"dist={d:.0f} -> click({ax},{ay})")

        if not DRY_RUN:
            win_input.click_abs(ax, ay)

        self.last_pos = target['click']
        self.recent.append((target['click'], now))
        self.clicks += 1

        if DRY_RUN:
            # клика нет -> глоу не появится; имитируем «срубку» паузой
            self.resume_at = now + DRY_WAIT
            self.state = SCAN
            return False

        # старт ожидания глоу
        self.t0 = now
        self.hard_deadline = now + HARD_CAP
        self.appear_deadline = min(now + APPEAR_BASE + d * WALK_SEC_PER_PX,
                                   now + APPEAR_MAX, self.hard_deadline)
        self.base = self._glow(bgr)   # фон в момент клика (перс не дошёл)
        self.peak = self.base
        self.last = self.base
        self.gone = 0
        self.state = WAIT_APPEAR
        return False

    def _tick_appear(self, now):
        bgr = grab_region(self.region)
        self.last = self._glow(bgr)
        self.peak = max(self.peak, self.last)
        if self.last >= self.base + NEON_RISE_PX:
            self.gone = 0
            self.state = WAIT_VANISH
            return
        if now >= self.appear_deadline:
            # глоу не появился -> куст/декор/недоступно -> чёрный список
            self.blacklist.append((self.target['click'], now))
            print(f"[{self.tag} #{self.clicks}] no_highlight "
                  f"(base={self.base} peak={self.peak}) -> чёрный список "
                  f"{self.target['click']}")
            self.state = SCAN

    def _tick_vanish(self, now):
        if now >= self.hard_deadline:
            print(f"[{self.tag} #{self.clicks}] timeout "
                  f"(base={self.base} peak={self.peak} last={self.last})")
            self.resume_at = now + random.uniform(*POST_CHOP_PAUSE)
            self.state = SCAN
            return
        bgr = grab_region(self.region)
        self.last = self._glow(bgr)
        self.peak = max(self.peak, self.last)
        if self.last <= self.base + NEON_FALL_PX:
            self.gone += 1
            if self.gone >= VANISH_CONFIRM:
                took = now - self.t0
                print(f"[{self.tag} #{self.clicks}] chopped за {took:.1f}s "
                      f"(base={self.base} peak={self.peak} last={self.last})")
                self.resume_at = now + random.uniform(*POST_CHOP_PAUSE)
                self.state = SCAN
        else:
            self.gone = 0

# ============================================================
# Планировщик: по кругу тикает 4 аккаунта, владеет единым курсором
# ============================================================

def main():
    regions = build_regions()
    print("=" * 56)
    print(" TREE BOT x4 (interleave, единый курсор)")
    print(f"  DRY_RUN = {DRY_RUN}")
    print(f"  Монитор = [{MONITOR_INDEX}]  DPI = {win_input.set_dpi_aware()}")
    print(f"  Виртуальный стол = {win_input.virtual_screen()}")
    for r in regions:
        print(f"    {r['name']}: left={r['left']} top={r['top']} "
              f"{r['width']}x{r['height']}  сервер={r['target_server']}")
    templates = load_templates(UI_DIR)
    if templates.get('_ocr') is None:
        print("  ВНИМАНИЕ: OCR недоступен — перезаход выключен (фарм работает).")
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
        time.sleep(0.01)   # лёгкий yield, не жжём CPU (троттл — внутри tick)

    total = sum(b.clicks for b in bots)
    print(f"Готово. Кликов всего: {total} "
          f"({', '.join(f'{b.tag}={b.clicks}' for b in bots)})")


if __name__ == "__main__":
    main()
