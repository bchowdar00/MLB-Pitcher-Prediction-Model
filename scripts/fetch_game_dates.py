"""
Fetch the official calendar date for every game_pk used by the dataset.

game_pk is NOT monotonic with date, so the model's "next game" target and
rolling features need the real date. This script reads the game_pk values from
``data/features/pitcher_games_enhanced.csv`` and queries MLB's public statsapi
schedule endpoint in batches, writing ``data/game_dates.csv`` with columns
``game_pk,date,game_type``.

The output CSV is committed so the evaluation reproduces offline; this script
only needs to be rerun if the source data changes. Uses the standard library
only (no extra dependencies).

    python scripts/fetch_game_dates.py
"""

import csv
import json
import os
import time
import urllib.request

HERE = os.path.dirname(__file__)
ENHANCED = os.path.join(HERE, "..", "data", "features", "pitcher_games_enhanced.csv")
OUT = os.path.join(HERE, "..", "data", "game_dates.csv")
ENDPOINT = "https://statsapi.mlb.com/api/v1/schedule?sportId=1&gamePks="
BATCH = 200


def read_game_pks(path):
    with open(path, newline="") as f:
        return sorted({int(row["game_pk"]) for row in csv.DictReader(f)})


def read_cache(path):
    if not os.path.exists(path):
        return {}
    with open(path, newline="") as f:
        return {int(r["game_pk"]): (r["date"], r["game_type"]) for r in csv.DictReader(f)}


def fetch_batch(pks):
    url = ENDPOINT + ",".join(str(p) for p in pks)
    with urllib.request.urlopen(url, timeout=60) as resp:
        data = json.load(resp)
    out = {}
    for day in data.get("dates", []):
        for g in day.get("games", []):
            out[int(g["gamePk"])] = (g["officialDate"], g["gameType"])
    return out


def main():
    pks = read_game_pks(ENHANCED)
    cache = read_cache(OUT)
    todo = [p for p in pks if p not in cache]
    print(f"{len(pks)} game_pks; {len(cache)} cached; {len(todo)} to fetch")

    for i in range(0, len(todo), BATCH):
        batch = todo[i : i + BATCH]
        cache.update(fetch_batch(batch))
        print(f"  fetched {min(i + BATCH, len(todo))}/{len(todo)}")
        time.sleep(0.5)

    rows = [(p, cache[p][0], cache[p][1]) for p in pks if p in cache]
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["game_pk", "date", "game_type"])
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {os.path.relpath(OUT)}")
    missing = [p for p in pks if p not in cache]
    if missing:
        print(f"WARNING: {len(missing)} game_pks had no schedule entry: {missing[:10]}")


if __name__ == "__main__":
    main()
