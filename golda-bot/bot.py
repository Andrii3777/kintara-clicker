import json
import logging
import logging.handlers
import os
import random
import signal
import threading
import time

import requests

from config import (
    API_FAIL_ALERT_AFTER, API_URL, ALLOWED_CHAT_IDS, BOT_TOKEN,
    NOTIFY_THRESHOLDS, POLL_INTERVAL, RESOURCES, STATE_FILE,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
_log_fmt = "%(asctime)s [%(levelname)s] %(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=_log_fmt,
    handlers=[
        logging.StreamHandler(),
        logging.handlers.RotatingFileHandler(
            "golda-bot.log", maxBytes=2_000_000, backupCount=3, encoding="utf-8"
        ),
    ],
)
log = logging.getLogger("golda-bot")

TG_BASE = f"https://api.telegram.org/bot{BOT_TOKEN}"

# ---------------------------------------------------------------------------
# Shared state + locks
# ---------------------------------------------------------------------------
_lock = threading.Lock()

# {(resource, threshold): True} — persisted to STATE_FILE
_notified: set[tuple[str, float]] = set()

_last_update_id = 0
_status_last: dict[int, float] = {}

_cache_data: dict | None = None
_cache_time: float = 0

_api_fail_count = 0
_api_fail_alerted = False

STATUS_COOLDOWN = 30
CACHE_TTL = 30

_stop = threading.Event()

# ---------------------------------------------------------------------------
# State persistence
# ---------------------------------------------------------------------------

def _load_state():
    global _notified
    if not os.path.exists(STATE_FILE):
        return
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            raw = json.load(f)
        with _lock:
            _notified = {(r, float(t)) for r, t in raw}
        log.info("State loaded: %d notified entries", len(_notified))
    except Exception as e:
        log.warning("Could not load state: %s", e)


def _save_state():
    try:
        with _lock:
            snapshot = list(_notified)
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(snapshot, f)
    except Exception as e:
        log.warning("Could not save state: %s", e)


# ---------------------------------------------------------------------------
# HTTP session
# ---------------------------------------------------------------------------
_session = requests.Session()
_session.headers.update({
    "accept": "*/*",
    "accept-language": "en-GB,en-US;q=0.9,en;q=0.8,ru;q=0.7",
    "cache-control": "no-cache",
    "pragma": "no-cache",
    "priority": "u=1, i",
    "referer": "https://kintara.gg/",
    "sec-ch-ua": '"Google Chrome";v="149", "Chromium";v="149", "Not)A;Brand";v="24"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"Windows"',
    "sec-fetch-dest": "empty",
    "sec-fetch-mode": "cors",
    "sec-fetch-site": "same-site",
    "user-agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/149.0.0.0 Safari/537.36"
    ),
})


def _fetch_from_api() -> dict:
    """One real HTTP request with 3-attempt exponential backoff."""
    global _api_fail_count, _api_fail_alerted
    last_exc = None
    for attempt in range(3):
        try:
            r = _session.get(API_URL, timeout=10)
            r.raise_for_status()
            with _lock:
                _api_fail_count = 0
                _api_fail_alerted = False
            return r.json()
        except Exception as e:
            last_exc = e
            if attempt < 2:
                time.sleep(2 ** attempt)

    with _lock:
        _api_fail_count += 1
        count = _api_fail_count
        alerted = _api_fail_alerted

    log.error("API unavailable (%d consecutive failures): %s", count, last_exc)

    if count >= API_FAIL_ALERT_AFTER and not alerted:
        with _lock:
            _api_fail_alerted = True
        send_message(f"⚠️ Kintara API unreachable for {count} polls in a row.")

    raise last_exc


def fetch_campaign(jitter: bool = True) -> dict:
    """Return cached data if fresh, otherwise fetch."""
    global _cache_data, _cache_time
    with _lock:
        if _cache_data and (time.time() - _cache_time) < CACHE_TTL:
            return _cache_data

    if jitter:
        time.sleep(random.uniform(0, 15))

    data = _fetch_from_api()

    with _lock:
        _cache_data = data
        _cache_time = time.time()

    return data


# ---------------------------------------------------------------------------
# Telegram
# ---------------------------------------------------------------------------

def send_message(text: str, chat_id: int | None = None):
    targets = [chat_id] if chat_id else list(ALLOWED_CHAT_IDS)
    for cid in targets:
        try:
            requests.post(
                f"{TG_BASE}/sendMessage",
                json={"chat_id": cid, "text": text, "parse_mode": "HTML"},
                timeout=10,
            )
        except Exception as e:
            log.warning("Failed to send message to %s: %s", cid, e)


def build_status(data: dict) -> str:
    lines = ["<b>Merchant Campaign Status</b>"]
    for res in RESOURCES:
        current = data.get(res, 0)
        goal = data["goals"].get(res, 1)
        pct = current / goal * 100
        bar = "🟩" * int(pct // 10) + "⬜" * (10 - int(pct // 10))
        lines.append(f"{bar} <b>{res}</b>: {current:,} / {goal:,} ({pct:.1f}%)")
    complete = "✅ COMPLETE" if data.get("complete") else "⏳ In progress"
    lines.append(f"\nCampaign: {complete}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Notification logic
# ---------------------------------------------------------------------------

def check_and_notify(data: dict):
    alerts = []
    with _lock:
        for res in RESOURCES:
            current = data.get(res, 0)
            goal = data["goals"].get(res, 1)
            pct = current / goal * 100

            for step in NOTIFY_THRESHOLDS:
                key = (res, step)
                if pct >= step and key not in _notified:
                    _notified.add(key)
                    emoji = "🏆" if step == 100 else ("🚨" if step >= 90 else "📊")
                    alerts.append(
                        f"{emoji} <b>{res.upper()}</b> {step}%"
                        f" — {current:,} / {goal:,} ({pct:.1f}%)"
                    )
                elif pct < step and key in _notified:
                    _notified.discard(key)

    if alerts:
        _save_state()
        send_message("\n".join(alerts))


# ---------------------------------------------------------------------------
# Poll loop
# ---------------------------------------------------------------------------

def poll_loop():
    log.info("Polling every %ds, steps=%s", POLL_INTERVAL, NOTIFY_THRESHOLDS)
    while not _stop.is_set():
        try:
            data = fetch_campaign(jitter=True)
            check_and_notify(data)
            for res in RESOURCES:
                current = data.get(res, 0)
                goal = data["goals"].get(res, 1)
                pct = current / goal * 100
                log.info("%s: %s/%s (%.1f%%)", res, f"{current:,}", f"{goal:,}", pct)
        except Exception as e:
            log.error("Poll error: %s", e)

        sleep_time = POLL_INTERVAL * random.uniform(0.8, 1.2)
        _stop.wait(sleep_time)


# ---------------------------------------------------------------------------
# Telegram update loop
# ---------------------------------------------------------------------------

def handle_updates():
    global _last_update_id
    log.info("Listening for Telegram commands...")
    while not _stop.is_set():
        try:
            r = requests.get(
                f"{TG_BASE}/getUpdates",
                params={"offset": _last_update_id + 1, "timeout": 30},
                timeout=35,
            )
            updates = r.json().get("result", [])
            for upd in updates:
                _last_update_id = upd["update_id"]
                msg = upd.get("message", {})
                text = msg.get("text", "")
                chat_id = msg.get("chat", {}).get("id")

                if text.strip() == "/status" and chat_id in ALLOWED_CHAT_IDS:
                    now = time.time()
                    with _lock:
                        since = now - _status_last.get(chat_id, 0)
                        if since < STATUS_COOLDOWN:
                            wait = int(STATUS_COOLDOWN - since)
                        else:
                            _status_last[chat_id] = now
                            wait = 0

                    if wait:
                        send_message(f"⏳ Wait {wait}s before next /status", chat_id=chat_id)
                    else:
                        try:
                            data = fetch_campaign(jitter=False)
                            send_message(build_status(data), chat_id=chat_id)
                        except Exception as e:
                            send_message(f"❌ Error fetching data: {e}", chat_id=chat_id)

        except Exception as e:
            log.error("Update loop error: %s", e)
            _stop.wait(5)


# ---------------------------------------------------------------------------
# Graceful shutdown
# ---------------------------------------------------------------------------

def _shutdown(signum, frame):
    log.info("Shutting down (signal %d)...", signum)
    _stop.set()
    _save_state()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    _load_state()

    t = threading.Thread(target=poll_loop, daemon=True)
    t.start()

    handle_updates()

    log.info("Bot stopped.")
