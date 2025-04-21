#!/usr/bin/env python3
import csv
import json
import time
import requests

# ——— CONFIG ———
CSV_PATH     = "top_1000_twitch.csv"
OUTPUT_JSON  = "twitchtracker.json"
BASE_API     = "https://twitchtracker.com/api"
DELAY        = 1.0     # seconds between *successful* requests
MAX_RETRIES  = 3       # for non-429 errors
BACKOFF_INIT = 1.0     # initial backoff on non-429 errors
BACKOFF_429  = 30.0    # how long to sleep on each 429 before retrying

def fetch_json(url):
    """
    GET + retry/backoff.
    - On 429: sleep BACKOFF_429 seconds, then retry forever.
    - On other network/5xx: retry up to MAX_RETRIES with exponential backoff.
    - On 4xx != 429: give up immediately.
    """
    # First, infinite loop for handling 429
    while True:
        try:
            resp = requests.get(url, timeout=10)
        except requests.RequestException as e:
            # network-level error: break to outer retry mechanism
            print(f"   ⚠️ Network error on {url}: {e}")
            break

        if resp.status_code == 429:
            print(f"   ‼️ 429 Too Many Requests on {url}, sleeping {BACKOFF_429}s")
            time.sleep(BACKOFF_429)
            continue

        # if not 429, break to handle below
        break

    # if we got here with a non-429 resp (or network error), handle retries
    backoff = BACKOFF_INIT
    for attempt in range(1, MAX_RETRIES + 1):
        if 'resp' not in locals():
            # network error case, try again
            pass
        else:
            # we have a response
            if 200 <= resp.status_code < 300:
                return resp.json()
            elif 400 <= resp.status_code < 500:
                print(f"   ❌ {resp.status_code} client error on {url}, giving up")
                return None
            else:
                # 5xx server error
                print(f"   ⚠️ {resp.status_code} server error on {url}, retry {attempt}/{MAX_RETRIES}")
        
        # wait then retry
        time.sleep(backoff)
        backoff *= 2
        try:
            resp = requests.get(url, timeout=10)
        except requests.RequestException as e:
            print(f"   ⚠️ Network retry error on {url}: {e}")
            continue

    # final check
    if 'resp' in locals() and 200 <= resp.status_code < 300:
        return resp.json()

    print(f"   ❌ Failed after retries on {url}")
    return None

def main():
    # read streamer slugs from CSV
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        streamers = [row["Name"].strip() for row in reader if row["Name"].strip()]

    all_data = {}
    for name in streamers:
        slug = name.lower()
        print(f"→ {name}")

        entry = {}

        # 1) Try channel summary first
        ch_url = f"{BASE_API}/channels/summary/{name}"
        ch_summary = fetch_json(ch_url)
        if ch_summary:
            entry["channel_summary_30d"] = ch_summary
            entry["games_summary_30d"]   = None
        else:
            # fallback to games summary
            print(f"   ℹ️ channel summary failed, trying games summary")
            gm_url = f"{BASE_API}/games/summary/{name}"
            gm_summary = fetch_json(gm_url)
            entry["channel_summary_30d"] = None
            entry["games_summary_30d"]   = gm_summary

        all_data[slug] = entry
        # only throttle *after* a successful fetch (or final failure)
        time.sleep(DELAY)

    # write out combined JSON
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)

    print(f"✅ Wrote {len(all_data)} entries into {OUTPUT_JSON}")

if __name__ == "__main__":
    main()
