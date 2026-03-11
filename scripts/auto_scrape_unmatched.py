#!/usr/bin/env python3
"""Headless Selenium crawler for unmatched players.

This script reads `data/mappings/player_map_review.csv` and for every row
with status 'unmatched' it will:

  * perform a Sofifa search by the `player_name_sb` value,
  * open the first result and extract the POINT_* attribute stats from the
    embedded JavaScript variables,
  * append a row containing the scraped stats to
    `data/mappings/scraped_attrs.csv`,
  * append a new accepted mapping to `data/mappings/player_map.csv` (method
    "scrape"),
  * update the review row with the found candidate id/name/score/method and
    set `status` to 'accepted_scrape',
  * optionally copy any newly‑scraped attributes into the players parquet
    table (`data/processed/players_train_1000.parquet`).

Usage:
    uv run scripts/auto_scrape_unmatched.py

Warning: network access to sofifa.com is required and a headless Chrome
instance will be launched.  Expect the run to take some time for many
players.
"""

from pathlib import Path
import pandas as pd
import sys
import time
import urllib.parse
import re  # for URL id extraction

# rapidfuzz for offline fuzzy matching
from rapidfuzz import process, fuzz

try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service
    from webdriver_manager.chrome import ChromeDriverManager
except ImportError:
    print("Please install selenium and webdriver-manager (uv add selenium webdriver-manager)")
    sys.exit(1)

# load FIFA cache for offline lookup
FIFA_CACHE_P = Path('data') / 'cache' / 'fifa_players.parquet'
if FIFA_CACHE_P.exists():
    _fifa_df = pd.read_parquet(FIFA_CACHE_P)
    # primary name field is `long_name` (full name); also keep short for display
    _candidate_names = _fifa_df['long_name'].astype(str).tolist()
else:
    _fifa_df = None
    _candidate_names = []

REVIEW_P = Path('data') / 'mappings' / 'player_map_review.csv'
MAP_P = Path('data') / 'mappings' / 'player_map.csv'
SCRAPED_P = Path('data') / 'mappings' / 'scraped_attrs.csv'
PLAYERS_P = Path('data') / 'processed' / 'players_train_1000.parquet'

YEAR_ROSTER = '230040'  # year token used in Sofifa URLs for FIFA23

# attribute variable names in JS
JS_STATS = {
    'pace': 'POINT_PAC',
    'shooting': 'POINT_SHO',
    'passing': 'POINT_PAS',
    'dribbling': 'POINT_DRI',
    'defending': 'POINT_DEF',
    'physic': 'POINT_PHY',
}


def init_driver():
    opts = Options()
    opts.add_argument('--headless=new')
    opts.add_argument('--disable-gpu')
    opts.add_argument('--no-sandbox')
    opts.add_argument('--disable-dev-shm-usage')
    opts.add_argument('--window-size=1920,1080')
    opts.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64)')
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=opts)
    return driver


def extract_stats_from_page(driver):
    # page_source may contain a <script> block with POINT_* variable assignments
    html = driver.page_source
    stats = {}
    for attr, var in JS_STATS.items():
        m = re.search(rf"{var}\s*=\s*(\d+)", html)
        stats[attr] = int(m.group(1)) if m else None
    return stats


def search_player(driver, name: str):
    """
    Lookup *name* and return a tuple ``(fifa_id, candidate_name, score)``.

    ``score`` is the fuzzy score for offline matches (0–100); it will be
    ``None`` for an online lookup.  ``fifa_id`` may also be ``None`` if the
    lookup fails.
    """
    # offline lookup
    if _fifa_df is not None and _candidate_names:
        # debug: show that offline dataset is available
        print(f'offline candidates count {len(_candidate_names)} for query {name}')
        res = process.extractOne(
            name,
            _candidate_names,
            scorer=fuzz.WRatio,
            score_cutoff=65,
            processor=lambda s: s
        )
        # debug
        print('offline lookup result', res)
        if res is not None:
            candidate, score, idx = res
            if candidate:
                if 'sofifa_id' in _fifa_df.columns:
                    fifa_id = int(_fifa_df.iloc[idx]['sofifa_id'])
                elif 'player_id' in _fifa_df.columns:
                    fifa_id = int(_fifa_df.iloc[idx]['player_id'])
                else:
                    fifa_id = None
                return fifa_id, candidate, score
    # fallback to web search
    q = urllib.parse.quote(name)
    url = f"https://sofifa.com/players?search={q}"
    driver.get(url)
    time.sleep(2)  # give time for JS to populate
    try:
        anchors = driver.find_elements('css selector', 'a')
        parts = [p.lower() for p in name.split() if p]
        for a in anchors:
            href = a.get_attribute('href') or ''
            if '/player/' not in href:
                continue
            text = a.text.strip().lower()
            if any(p in text for p in parts):
                m = re.search(r"/player/(\d+)", href)
                if m:
                    return int(m.group(1)), a.text.strip(), None
        for a in anchors:
            href = a.get_attribute('href') or ''
            if '/player/' in href:
                m = re.search(r"/player/(\d+)", href)
                if m:
                    return int(m.group(1)), a.text.strip(), None
        return None, None, None
    except Exception as e:
        print("  search parse failed:", e)
        return None, None, None


def main():
    if not REVIEW_P.exists():
        print('review file not found', REVIEW_P)
        sys.exit(1)
    review = pd.read_csv(REVIEW_P, dtype={'player_id_sb': object,
                                          'player_name_sb': object,
                                          'candidate_fifa_id': object,
                                          'status': object})
    unmatched = review[review['status'] == 'unmatched'].copy()
    print(f'Processing {len(unmatched)} unmatched rows')
    if unmatched.empty:
        return

    driver = init_driver()
    scraped_rows = []
    map_rows = []

    for idx, r in unmatched.iterrows():
        name = r['player_name_sb']
        print('searching', name)
        fid, cand_name, score = search_player(driver, name)
        if fid:
            player_url = f"https://sofifa.com/player/{fid}/{YEAR_ROSTER}"
            print('  candidate', fid, cand_name, 'score', score, '->', player_url)
            try:
                driver.get(player_url)
                time.sleep(1)
                stats = extract_stats_from_page(driver)
                print('  stats:', stats)
                scraped_rows.append({
                    'player_id_sb': r['player_id_sb'],
                    'player_name_sb': name,
                    'candidate_fifa_id': fid,
                    **stats,
                })
                map_rows.append({
                    'player_id_sb': r['player_id_sb'],
                    'player_name_sb': name,
                    'fifa_id': fid,
                    'score': score if score is not None else 0.0,
                    'method': 'scrape',
                })
                review.at[idx, 'candidate_fifa_id'] = fid
                review.at[idx, 'candidate_name'] = cand_name
                review.at[idx, 'score'] = score if score is not None else 0.0
                review.at[idx, 'method'] = 'scrape'
                review.at[idx, 'status'] = 'review'
            except Exception as e:
                print('  error fetching player page', e)
        else:
            print('  no candidate found for', name)

    driver.quit()

    if scraped_rows:
        df = pd.DataFrame(scraped_rows)
        if SCRAPED_P.exists():
            df.to_csv(SCRAPED_P, mode='a', header=False, index=False)
        else:
            df.to_csv(SCRAPED_P, index=False)
        print('Wrote scraped attributes to', SCRAPED_P)

    if map_rows:
        dfm = pd.DataFrame(map_rows)
        if MAP_P.exists():
            dfm.to_csv(MAP_P, mode='a', header=False, index=False)
        else:
            dfm.to_csv(MAP_P, index=False)
        print('Appended new mappings to', MAP_P)

    review.to_csv(REVIEW_P, index=False)
    print('Updated review file', REVIEW_P)

    try:
        players = pd.read_parquet(PLAYERS_P)
        for r in scraped_rows:
            idxs = players[players['player_name_sb'] == r['player_name_sb']].index
            for i in idxs:
                for a in JS_STATS.keys():
                    val = r.get(a)
                    if pd.isna(players.at[i, a]) and val is not None:
                        players.at[i, a] = val
        players.to_parquet(PLAYERS_P, index=False)
        print('Copied scraped attributes into players_train_1000.parquet')
    except Exception as e:
        print('failed to update players table', e)


if __name__ == '__main__':
    main()
