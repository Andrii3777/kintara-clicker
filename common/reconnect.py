# -*- coding: utf-8 -*-
"""
reconnect.py
============
Слой ПЕРЕЗАХОДА на сервер. ОБЩИЙ для всех ботов (как win_input).

ЗРЕНИЕ перезахода через OCR (RapidOCR / onnxruntime): читаем ТЕКСТ кнопок и
строк меню, а не пиксель-точные шаблоны. Масштаб/разрешение/монитор-независимо.
Сам клик/скролл делает бот (через win_input). Тут — ТОЛЬКО зрение.

РЕАЛЬНЫЙ ФЛОУ ИГРЫ (по debug-скринам):
  1. "SELECT A SERVER" / "CHOOSE YOUR REALM" — список миров. Скроллить.
     - KINTARA CLUB N  — "MEMBERS ONLY" платный → ПРОПУСКАЕМ (нет queue/instant).
     - SERVER N        — бесплатный:
         "OPEN · JOIN INSTANTLY"  → приоритет 1, заходим сразу.
         "FULL · X IN QUEUE"      → берём наименьший X если открытых нет.
  2. Выбрали → "YOU ARE IN QUEUE" (players ahead) → ЖДЁМ, ничего не жмём.
  3. В очереди может выскочить "SOMETHING WENT WRONG" / "CONNECTION ERROR" →
     жмём RETRY → заново выбираем сервер.

Приоритет emergency_ui_check (сверху вниз):
  RETRY/ошибка → click RETRY
  очередь (players ahead / you are in queue) → wait
  loading/connecting → wait
  OK (попап вылета) → click
  Play Now (главное меню) → click
  список серверов → выбор открытого / наименьшей очереди / скролл

СОСТОЯНИЕ: выбор сервера требует памяти между кадрами (скролл по списку в поисках
открытого, детект низа). Бот передаёт per-account dict `state`; reconnect его ведёт.
Нет state → выбор всё равно работает (скроллит ища instant), но не «коммитит»
наименьшую очередь (нет памяти низа).

Координаты в действиях — ЛОКАЛЬНЫЕ внутри региона (бот добавит offset).
scroll-действие несёт focus_x/focus_y (безопасная точка фокуса — заголовок, НЕ
карточка) + x/y (точка колеса над списком).
"""

import re
import cv2

MIN_CONF = 0.50          # минимальная уверенность OCR
MAX_SCROLLS = 15         # потолок скроллов списка (страховка от джиттера OCR)
OCR_UPSCALE_TO = 1600    # кадр уже этого по ширине -> апскейл (мелкий текст x4)

# Зоны-мусор (доли кадра), где OCR ловит НЕ попапы: верх браузера (вкладки/адрес/
# закладки — "kintara.gg/play"), чат снизу-слева ("donate now"), таскбар. Попапы
# перезахода/очереди/сервера ВСЕГДА по центру -> эти зоны игнорируем.
# (x0, y0, x1, y1) в долях. Подгони под свой браузер если хром выше/ниже.
IGNORE_ZONES = [
    (0.00, 0.00, 1.00, 0.14),   # верх: вкладки + адресная строка + закладки
    (0.00, 0.60, 0.36, 1.00),   # низ-лево: окно чата
    (0.00, 0.95, 1.00, 1.00),   # низ: таскбар Windows
]

# "FULL · 20 IN QUEUE" / "FULL·15INQUEUE" / "FULL-12IN QUEUE" -> число очереди
QUEUE_RE = re.compile(r"(\d+)\s*IN\s*QUEUE")

_OCR = None
_OCR_FAILED = False


def _get_ocr():
    """Синглтон RapidOCR. Нет пакета -> None (перезаход выключится, фарм живёт)."""
    global _OCR, _OCR_FAILED
    if _OCR is not None or _OCR_FAILED:
        return _OCR
    try:
        from rapidocr_onnxruntime import RapidOCR
        _OCR = RapidOCR()
    except Exception as e:
        _OCR_FAILED = True
        print(f"  [reconnect] OCR недоступен ({e}) — перезаход отключён. "
              f"pip install rapidocr_onnxruntime")
    return _OCR


def load_templates(ui_dir=None):
    """
    Совместимость со старым контрактом ботов (templates.items(), missing-check).
    Возвращает dict {'_ocr': engine}. ui_dir не используется (OCR не нужны PNG).
    """
    return {"_ocr": _get_ocr()}


# --- OCR --------------------------------------------------------------------
def _box_center(box):
    xs = [p[0] for p in box]
    ys = [p[1] for p in box]
    return (int(sum(xs) / 4), int(sum(ys) / 4))


def _run_ocr(engine, frame):
    """OCR -> список (text_upper, center, conf). Кормить ЦВЕТНОЙ кадр (gray
    теряет текст на цветных кнопках)."""
    if engine is None:
        return []
    img = (cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
           if frame.ndim == 2 else frame)
    # АПСКЕЙЛ мелких кадров (x4 квадрант ~640px): иначе OCR не читает строки
    # серверов (текст ~10px). Координаты делим обратно на scale.
    h, w = img.shape[:2]
    scale = 1.0
    if w < OCR_UPSCALE_TO:
        scale = OCR_UPSCALE_TO / w
        img = cv2.resize(img, (int(w * scale), int(h * scale)),
                         interpolation=cv2.INTER_CUBIC)
    result, _ = engine(img)
    if not result:
        return []
    out = []
    for box, text, conf in result:
        cx, cy = _box_center(box)
        out.append((text.upper().strip(),
                    (int(cx / scale), int(cy / scale)), float(conf)))
    return out


def _find(items, *keywords, min_conf=MIN_CONF):
    """Первый бокс, содержащий ЛЮБОЕ keyword (подстрока). center или None."""
    for text, center, conf in items:
        if conf >= min_conf and any(k in text for k in keywords):
            return center
    return None


def _find_multi(items, keywords, min_conf=MIN_CONF):
    """Кросс-бокс: каждое keyword в КАКОМ-ТО боксе (OCR дробит фразы). center
    бокса с первым keyword, иначе None."""
    centers = {}
    for text, center, conf in items:
        if conf < min_conf:
            continue
        for k in keywords:
            if k in text:
                centers.setdefault(k, center)
    if all(k in centers for k in keywords):
        return centers[keywords[0]]
    return None


def _in_ignore(center, W, H):
    """Точка в зоне-мусоре (браузер/чат/таскбар)? -> True (игнорировать)."""
    fx, fy = center[0] / W, center[1] / H
    return any(x0 <= fx <= x1 and y0 <= fy <= y1
               for x0, y0, x1, y1 in IGNORE_ZONES)


def _find_lowest(items, *keywords, min_conf=MIN_CONF):
    """Из боксов с ЛЮБЫМ keyword вернуть САМЫЙ НИЖНИЙ (max y) center, иначе None.
    Для попапов где фраза есть и в тексте, и на кнопке — кнопка внизу."""
    hit = [center for text, center, conf in items
           if conf >= min_conf and any(k in text for k in keywords)]
    return max(hit, key=lambda c: c[1]) if hit else None


# --- Парс списка серверов ---------------------------------------------------
def _parse_servers(items):
    """
    Бесплатные сервера из видимого списка. Платные (KINTARA CLUB / MEMBERS ONLY)
    отсекаются сами: у них нет строки "IN QUEUE"/"INSTANTLY".
    Вернуть [{'q':int(0=open), 'instant':bool, 'center':(x,y)}].
    """
    servers = []
    for text, center, conf in items:
        if conf < MIN_CONF:
            continue
        if "INSTANTLY" in text:
            servers.append({"q": 0, "instant": True, "center": center})
            continue
        m = QUEUE_RE.search(text)
        if m:
            servers.append({"q": int(m.group(1)), "instant": False,
                            "center": center})
    return servers


def _server_select(items, state):
    """Решение на экране выбора сервера. dict действия (click/scroll)."""
    st = state if isinstance(state, dict) else {}

    # заголовок = безопасная точка фокуса для скролла (НЕ карточка)
    title = _find(items, "SELECT A SERVER", "SELECT", "REALM")

    servers = _parse_servers(items)

    # 1) Открытый сервер виден -> заходим сразу (нет очереди).
    instant = [s for s in servers if s["instant"]]
    if instant:
        st.clear()
        c = instant[0]["center"]
        return {"action": "click", "x": c[0], "y": c[1],
                "msg": "Открытый сервер (JOIN INSTANTLY) — захожу."}

    # точка колеса — над списком (центр видимых строк)
    pts = ([s["center"] for s in servers]
           or [c for t, c, cf in items
               if cf >= MIN_CONF and (title is None or c[1] > title[1] + 40)])
    if pts:
        xs = sorted(p[0] for p in pts)
        ys = sorted(p[1] for p in pts)
        wheel = (xs[len(xs) // 2], ys[len(ys) // 2])
    else:
        wheel = title or (0, 0)

    # 2) Открытых нет. Детект низа: сигнатура очередей не меняется после скролла
    #    ИЛИ исчерпан лимит скроллов -> коммитим наименьшую очередь.
    sig = tuple(sorted(s["q"] for s in servers))
    scrolls = st.get("scrolls", 0)
    bottom = servers and (sig == st.get("prev_sig") or scrolls >= MAX_SCROLLS)
    if bottom:
        st.clear()
        best = min(servers, key=lambda s: s["q"])
        c = best["center"]
        return {"action": "click", "x": c[0], "y": c[1],
                "msg": f"Открытых нет — беру наименьшую очередь ({best['q']})."}

    # 3) Иначе мотаем список дальше (ищем открытый / доходим до низа).
    st["prev_sig"] = sig if servers else st.get("prev_sig")
    st["scrolls"] = scrolls + 1
    act = {"action": "scroll", "x": wheel[0], "y": wheel[1],
           "msg": "Список серверов — мотаю искать открытый."}
    if title:
        act["focus_x"], act["focus_y"] = title  # фокус на заголовок, не карточку
    return act


def emergency_ui_check(frame, target_server=None, templates=None, state=None):
    """
    Кадр (ЛУЧШЕ цветной BGR) -> dict действия перезахода или None.
    target_server — НЕ используется (оставлен для совместимости). state — per-account
    dict памяти для выбора сервера (см. модульный docstring).

    Действия (координаты ЛОКАЛЬНЫЕ внутри региона):
      {'action':'wait'}
      {'action':'click','x','y'}
      {'action':'scroll','x','y'[, 'focus_x','focus_y']}
    """
    engine = templates.get("_ocr") if isinstance(templates, dict) else _get_ocr()
    all_items = _run_ocr(engine, frame)   # полный список до фильтрации (нужен для Play Now)
    if not all_items:
        return None
    # выкинуть текст из зон-мусора (браузер/чат/таскбар) — попапы только по центру
    H, W = frame.shape[:2]
    items = [it for it in all_items if not _in_ignore(it[1], W, H)]
    if not items:
        return None
    joined = " ".join(t for t, _, c in items if c >= MIN_CONF)

    # 1) Ошибка коннекта -> RETRY (ВЫШЕ очереди: текст ошибки содержит "queue").
    if (_find(items, "RETRY", min_conf=0.40)
            or _find_multi(items, ("WENT", "WRONG"))
            or _find_multi(items, ("CONNECTION", "ERROR"))):
        pos = _find(items, "RETRY", min_conf=0.40)
        if pos:
            if isinstance(state, dict):
                state.clear()
            return {"action": "click", "x": pos[0], "y": pos[1],
                    "msg": "Ошибка коннекта — жму RETRY."}

    # 2) В очереди -> ждать (специфичные фразы, не путать со списком "IN QUEUE").
    if any(k in joined for k in ("PLAYERS AHEAD", "YOU ARE IN",
                                 "ENTER AUTOMATICALLY")):
        if isinstance(state, dict):
            state.clear()
        return {"action": "wait", "msg": "В очереди — жду."}

    # 3) Загрузка / коннект -> ждать.
    if any(k in joined for k in ("LOADING", "CONNECTING", "STARTING")):
        return {"action": "wait", "msg": "Загрузка/коннект — жду."}

    # 4) Попап вылета -> OK (точный токен, чат "OKOK" не считается).
    for text, center, conf in items:
        if conf >= MIN_CONF and text in ("OK", "ОК"):
            if isinstance(state, dict):
                state.clear()
            return {"action": "click", "x": center[0], "y": center[1],
                    "msg": "Вылет — жму OK."}

    # 4b) Попап торговца «Safe travels.» — клик кнопки (фраза есть и в теле
    #     текста, и на кнопке -> берём САМУЮ НИЖНЮЮ = кнопку).
    pos = _find_lowest(items, "TRAVELS")
    if pos:
        if isinstance(state, dict):
            state.clear()
        return {"action": "click", "x": pos[0], "y": pos[1],
                "msg": "Торговец ушёл — жму Safe travels."}

    # 5) Главное меню / профиль -> Play Now.
    # Порядок: 1) оба слова в ОДНОМ боксе (надёжнее), 2) кросс-бокс в отфильтрованных,
    # 3) одиночный бокс в НЕотфильтрованных (страховка: кнопка у края зоны чата
    # при высоком браузере).
    _play_now_pos = None
    for _text, _center, _conf in items:
        if _conf >= MIN_CONF and "PLAY" in _text and "NOW" in _text:
            _play_now_pos = _center
            break
    if _play_now_pos is None:
        _play_now_pos = _find_multi(items, ("PLAY", "NOW"))
    if _play_now_pos is None:
        for _text, _center, _conf in all_items:
            if _conf >= MIN_CONF and "PLAY" in _text and "NOW" in _text:
                _play_now_pos = _center
                break
    if _play_now_pos:
        if isinstance(state, dict):
            state.clear()
        return {"action": "click", "x": _play_now_pos[0], "y": _play_now_pos[1],
                "action_type": "play_now",
                "msg": "Главное меню — жму Play Now."}

    # 5b) Случайно зашли на платный (KINTARA CLUB / MEMBERS ONLY / PAID IN SOL) ->
    #     кнопка "Back to servers". ВЫШЕ выбора сервера: экран подписки содержит
    #     "CHOOSE A PLAN ... REALM" -> иначе ложно считается списком серверов.
    #     Требуем оба слова В ОДНОМ боксе ("BACK TO SERVERS") — не ник игрока "Back".
    for text, center, conf in items:
        if conf >= MIN_CONF and "BACK" in text and "SERVER" in text:
            if isinstance(state, dict):
                state.clear()
            return {"action": "click", "x": center[0], "y": center[1],
                    "msg": "Платный сервер — жму Back to servers."}

    # 6) Экран выбора сервера. Заголовок строго ДВУМЯ словами вместе
    #    ("SELECT A SERVER"/"CHOOSE YOUR REALM") ИЛИ распознаны строки серверов.
    #    НЕ голый "SERVER"/"REALM": в игре есть "Server 13" и чат-кнопка "Realm".
    if (_find_multi(items, ("SELECT", "SERVER"))
            or _find_multi(items, ("CHOOSE", "REALM"))
            or _parse_servers(items)):
        return _server_select(items, state)

    return None
