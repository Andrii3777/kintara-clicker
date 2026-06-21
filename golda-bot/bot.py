import time
import random
import requests
import threading
from config import BOT_TOKEN, CHAT_ID, API_URL, THRESHOLD, POLL_INTERVAL, RESOURCES

TG_BASE = f"https://api.telegram.org/bot{BOT_TOKEN}"

# Track which resources already triggered notification (edge-trigger)
_notified = set()
_last_update_id = 0

# Persistent session — reuses TCP connection, keeps cookies
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


def fetch_campaign():
    # Jitter ±15s so requests don't land on exact clock ticks
    time.sleep(random.uniform(0, 15))
    r = _session.get(API_URL, timeout=10)
    r.raise_for_status()
    return r.json()


def send_message(text):
    requests.post(f"{TG_BASE}/sendMessage", json={
        "chat_id": CHAT_ID,
        "text": text,
        "parse_mode": "HTML",
    }, timeout=10)


def build_status(data):
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


def check_and_notify(data):
    alerts = []
    for res in RESOURCES:
        current = data.get(res, 0)
        goal = data["goals"].get(res, 1)
        pct = current / goal

        if pct >= THRESHOLD and res not in _notified:
            _notified.add(res)
            alerts.append(
                f"🚨 <b>{res.upper()}</b> reached {pct*100:.1f}%! "
                f"({current:,} / {goal:,})"
            )
        elif pct < THRESHOLD and res in _notified:
            # Reset so we notify again if it dips and rises
            _notified.discard(res)

    if alerts:
        send_message("\n".join(alerts))


def poll_loop():
    print(f"[golda-bot] polling every {POLL_INTERVAL}s, threshold={THRESHOLD*100:.0f}%")
    while True:
        try:
            data = fetch_campaign()
            check_and_notify(data)
            # Print current status to console
            for res in RESOURCES:
                current = data.get(res, 0)
                goal = data["goals"].get(res, 1)
                pct = current / goal * 100
                print(f"  {res}: {current:,}/{goal:,} ({pct:.1f}%)")
        except Exception as e:
            print(f"[poll error] {e}")
        # Randomize interval ±20% to avoid fixed fingerprint
        sleep_time = POLL_INTERVAL * random.uniform(0.8, 1.2)
        time.sleep(sleep_time)


def handle_updates():
    global _last_update_id
    while True:
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
                if text.strip() == "/status" and chat_id == CHAT_ID:
                    try:
                        data = fetch_campaign()
                        send_message(build_status(data))
                    except Exception as e:
                        send_message(f"Error fetching data: {e}")
        except Exception as e:
            print(f"[update error] {e}")
            time.sleep(5)


if __name__ == "__main__":
    t = threading.Thread(target=poll_loop, daemon=True)
    t.start()
    print("[golda-bot] listening for /status commands...")
    handle_updates()
