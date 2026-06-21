# -*- coding: utf-8 -*-
"""
pan.py
======
Панорамирование карты — ОБЩЕЕ для всех ботов (как win_input/reconnect).

Целей нет (деревья/камень/уголь кончились) → обходим карту «газонокосилкой» по
ДИАГОНАЛЯМ ромба (изометрия). Всё по СЧЁТЧИКАМ, без детекта края:
  PAN_ROW_LEN панов вдоль ряда (ось row) -> шаг на соседний ряд (ось step) ->
  обратный ряд -> ... после PAN_NUM_ROWS шагов -> разворот оси step.
Заезд в пустоту безвреден (целей там нет). MAX_EMPTY_PANS подряд -> карта обойдена.

Состояние снейка инкапсулировано в `Panner` (раньше дублировалось в 4 ботах:
nonlocal в single, атрибуты в x4). Drag — через win_input из ЦЕНТРА региона
(монитор для single, квадрант для x4 — оба передают region с left/top/width/height).

Сброс фарм-памяти (last_pos/recent) и смену FSM-состояния делает БОТ, не Panner.
"""

import time
import cv2
import numpy as np

import win_input

# Диагональные направления (вдоль осей ромба) в экранных (dx, dy).
DIR_VEC = {'NE': (1, -1), 'SW': (-1, 1), 'NW': (-1, -1), 'SE': (1, 1)}
DIR_FLIP = {'NE': 'SW', 'SW': 'NE', 'NW': 'SE', 'SE': 'NW'}

# Значения по умолчанию (бот может переопределить через cfg).
DEFAULTS = {
    'PAN_DIST': 380,        # длина drag по каждой оси, px (диагональ ~*1.4)
    'PAN_DURATION': 0.6,    # время одного drag, сек
    'PAN_SETTLE': 0.6,      # пауза после drag чтоб карта устаканилась, сек
    'PAN_ROW_LEN': 12,      # панов вдоль ряда (ось SW<->NE)
    'PAN_NUM_ROWS': 20,     # шагов вдоль оси SE<->NW
    'ROW_START': 'SW',      # ось ряда: 'SW'(↙) <-> 'NE'(↗)
    'STEP_START': 'SE',     # ось шага: 'SE'(↘) <-> 'NW'(↖)
    'DRY_WAIT': 0.5,        # пауза вместо drag в DRY_RUN, сек
}


def frame_change(a, b):
    """Доля заметно изменившихся пикселей между кадрами a и b (0..1). Для лога."""
    diff = cv2.absdiff(a, b)
    return float(np.count_nonzero(diff.sum(2) > 30)) / (a.shape[0] * a.shape[1])


class Panner:
    """
    Обход карты змейкой. Один вызов step() = ОДИН пан (drag). Возвращает True
    когда вся карта обойдена (MAX_EMPTY_PANS панов без целей) — бот реагирует
    (single: стоп, x4: state=DONE).

    region : dict {'left','top','width','height'} — drag из его центра.
    cfg    : переопределения DEFAULTS (PAN_DIST, PAN_ROW_LEN, ...).
    grab   : callable() -> bgr-кадр для замера change% в логе (опц.). Нет -> без %.
    """

    def __init__(self, region, cfg=None, *, dry_run=False, tag="",
                 grab=None, log=print):
        self.region = region
        self.c = dict(DEFAULTS)
        if cfg:
            self.c.update({k: v for k, v in cfg.items() if k in self.c})
        self.dry_run = dry_run
        self.tag = tag
        self.grab = grab
        self.log = log
        # обойти всю карту: оба прохода рядов в обе стороны + запас
        self.max_empty = 2 * self.c['PAN_NUM_ROWS'] * (self.c['PAN_ROW_LEN'] + 1) + 8
        self.reset()

    def reset(self):
        """Сброс снейка (например после перезахода — карта/координаты другие)."""
        self.empty_pans = 0
        self.row_dir = self.c['ROW_START']
        self.step_dir = self.c['STEP_START']
        self.row_pans = 0
        self.step_count = 0
        self.pending_step = False

    def _drag(self, direction, before=None):
        """Drag из центра региона по диагонали. Вернуть change% (или -1 без grab)."""
        if self.dry_run:
            time.sleep(self.c['DRY_WAIT'])
            return -1.0
        r = self.region
        sx = r['left'] + r['width'] // 2
        sy = r['top'] + r['height'] // 2
        vx, vy = DIR_VEC[direction]
        d = self.c['PAN_DIST']
        win_input.drag_abs(sx, sy, sx + vx * d, sy + vy * d,
                           steps=25, duration=self.c['PAN_DURATION'])
        time.sleep(self.c['PAN_SETTLE'])
        if self.grab is not None and before is not None:
            return frame_change(before, self.grab())
        return -1.0

    def step(self, reason="", before=None):
        """
        Один пан. Вернуть True если карта обойдена (целей нет нигде) -> стоп/DONE.
        before — кадр ДО пана (для лога change%, опц.).
        """
        self.empty_pans += 1
        if self.empty_pans > self.max_empty:
            self.log(f"[{self.tag}] {self.max_empty} панов без целей — "
                     f"карта обойдена.")
            return True

        rl, nr = self.c['PAN_ROW_LEN'], self.c['PAN_NUM_ROWS']
        if self.pending_step:
            direction = self.step_dir
            ch = self._drag(direction, before)
            self.step_count += 1
            self.log(f"[{self.tag}] {reason} -> шаг '{direction}'{_pct(ch)} "
                     f"ряд {self.step_count}/{nr} "
                     f"({self.empty_pans}/{self.max_empty})")
            if self.step_count >= nr:
                self.step_count = 0
                self.step_dir = DIR_FLIP[self.step_dir]
            self.pending_step = False
        else:
            direction = self.row_dir
            ch = self._drag(direction, before)
            self.row_pans += 1
            self.log(f"[{self.tag}] {reason} -> пан '{direction}'{_pct(ch)} "
                     f"ряд {self.row_pans}/{rl} "
                     f"({self.empty_pans}/{self.max_empty})")
            if self.row_pans >= rl:
                self.row_pans = 0
                self.pending_step = True
                self.row_dir = DIR_FLIP[self.row_dir]
        return False


def _pct(ch):
    return f" измен={ch*100:.0f}%" if ch >= 0 else ""
