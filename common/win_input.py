# -*- coding: utf-8 -*-
"""
win_input.py
============
Надёжный клик на Windows для МУЛЬТИМОНИТОРА с любым DPI-масштабом.

Зачем не pyautogui:
  pyautogui кликает в "логических" координатах одного экрана и ломается на
  втором мониторе (особенно с отрицательным смещением, left=-1920) и при
  масштабе != 100%.

Решение:
  1) Делаем процесс DPI-aware -> метрики и mss-пиксели становятся в ОДНОЙ
     системе (физические пиксели). Сдвиг от масштаба пропадает.
  2) Кликаем через SendInput с MOUSEEVENTF_ABSOLUTE | VIRTUALDESK — координаты
     нормируются по ВСЕМУ виртуальному столу (0..65535), включая мониторы
     слева с отрицательным left. Отрицательные координаты больше не проблема.

Координаты на вход click_abs(x, y) — ФИЗИЧЕСКИЕ пиксели виртуального стола
(т.е. cx + mon['left'], cy + mon['top'] из mss).
"""

import ctypes
from ctypes import wintypes


# ---- DPI awareness (вызвать ОДИН раз в самом начале программы) ----
def set_dpi_aware():
    """Сделать процесс DPI-aware. Иначе метрики экрана врут при масштабе !=100%."""
    try:
        # Per-Monitor v2 (Win10+), лучший вариант
        ctypes.windll.user32.SetProcessDpiAwarenessContext(
            ctypes.c_void_p(-4))  # DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2
        return "per-monitor-v2"
    except Exception:
        pass
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)  # PER_MONITOR_AWARE
        return "per-monitor"
    except Exception:
        pass
    try:
        ctypes.windll.user32.SetProcessDPIAware()       # system-aware (старый)
        return "system"
    except Exception:
        return "none"


# ---- Метрики виртуального стола ----
SM_XVIRTUALSCREEN = 76
SM_YVIRTUALSCREEN = 77
SM_CXVIRTUALSCREEN = 78
SM_CYVIRTUALSCREEN = 79


def virtual_screen():
    """(left, top, width, height) всего виртуального стола в физ. пикселях."""
    gsm = ctypes.windll.user32.GetSystemMetrics
    return (gsm(SM_XVIRTUALSCREEN), gsm(SM_YVIRTUALSCREEN),
            gsm(SM_CXVIRTUALSCREEN), gsm(SM_CYVIRTUALSCREEN))


# ---- SendInput ----
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_WHEEL = 0x0800
MOUSEEVENTF_ABSOLUTE = 0x8000
MOUSEEVENTF_VIRTUALDESK = 0x4000
INPUT_MOUSE = 0
WHEEL_DELTA = 120          # один «щелчок» колеса


class _MOUSEINPUT(ctypes.Structure):
    _fields_ = [("dx", wintypes.LONG), ("dy", wintypes.LONG),
                ("mouseData", wintypes.DWORD), ("dwFlags", wintypes.DWORD),
                ("time", wintypes.DWORD),
                ("dwExtraInfo", ctypes.POINTER(wintypes.ULONG))]


class _INPUT(ctypes.Structure):
    class _U(ctypes.Union):
        _fields_ = [("mi", _MOUSEINPUT)]
    _anonymous_ = ("u",)
    _fields_ = [("type", wintypes.DWORD), ("u", _U)]


def _send(flags, nx=0, ny=0, data=0):
    inp = _INPUT(type=INPUT_MOUSE,
                 mi=_MOUSEINPUT(dx=nx, dy=ny, mouseData=data,
                                dwFlags=flags, time=0, dwExtraInfo=None))
    ctypes.windll.user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(inp))


def _to_abs(x, y):
    """Физ. пиксель -> нормированные 0..65535 по виртуальному столу."""
    vx, vy, vw, vh = virtual_screen()
    nx = int(round((x - vx) * 65535 / max(vw - 1, 1)))
    ny = int(round((y - vy) * 65535 / max(vh - 1, 1)))
    return nx, ny


def move_abs(x, y):
    nx, ny = _to_abs(x, y)
    _send(MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_VIRTUALDESK,
          nx, ny)


def click_abs(x, y):
    """Переместить курсор и кликнуть ЛКМ в точке (x,y) виртуального стола."""
    move_abs(x, y)
    nx, ny = _to_abs(x, y)
    base = MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_VIRTUALDESK
    _send(MOUSEEVENTF_LEFTDOWN | base, nx, ny)
    _send(MOUSEEVENTF_LEFTUP | base, nx, ny)


def scroll(amount, x=None, y=None):
    """
    Прокрутка колеса мыши. amount в «щелчках»: >0 вверх, <0 вниз
    (как pyautogui.scroll). Если заданы x,y — сперва переместить курсор туда
    (физ. пиксели виртуального стола), колесо шлёт событие в точку под курсором.
    """
    if x is not None and y is not None:
        move_abs(x, y)
    _send(MOUSEEVENTF_WHEEL, data=int(amount) * WHEEL_DELTA)


def drag_abs(x1, y1, x2, y2, steps=25, duration=0.6):
    """
    Перетягивание ЛКМ из (x1,y1) в (x2,y2) — для панорамирования карты.
    Зажать -> плавно вести (steps шагов за duration сек) -> отпустить.
    Плавность важна: резкий телепорт игра может не распознать как drag.
    Координаты — физ. пиксели виртуального стола.
    """
    import time
    base = MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_VIRTUALDESK
    move_abs(x1, y1)
    nx, ny = _to_abs(x1, y1)
    _send(MOUSEEVENTF_LEFTDOWN | base, nx, ny)
    time.sleep(0.05)
    for i in range(1, steps + 1):
        t = i / steps
        x = int(x1 + (x2 - x1) * t)
        y = int(y1 + (y2 - y1) * t)
        nx, ny = _to_abs(x, y)
        _send(MOUSEEVENTF_MOVE | base, nx, ny)
        time.sleep(duration / steps)
    nx, ny = _to_abs(x2, y2)
    _send(MOUSEEVENTF_LEFTUP | base, nx, ny)
