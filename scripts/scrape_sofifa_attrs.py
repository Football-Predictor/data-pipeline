#!/usr/bin/env python3
"""Crawl Sofifa pages with headless Chrome to retrieve FIFA23 attribute stats.

This script reads `data/mappings/player_map_review.csv` and for each row
with status 'unmatched' or 'review' and a numeric `candidate_fifa_id`, it
visits the corresponding Sofifa player page (year 230040) and extracts the
POINT_PAC, POINT_SHO, POINT_PAS, POINT_DRI, POINT_DEF, POINT_PHY values from
embedded JavaScript variables.

The results are appended to `data/mappings/scraped_attrs.csv` (player info +
stats). Optionally, the script can also append new rows to
`data/cache/fifa_players_1000.parquet` and copy the attributes into
`data/processed/players_train_1000.parquet` for immediate use.

Usage:
    uv run scripts/scrape_sofifa_attrs.py

Warning: this will launch a headless Chrome instance and may take a while
for many players. Network access to sofifa.com is required.
"""

from pathlib import Path
import pandas as pd
import sys
import time

try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service
    from webdriver_manager.chrome import ChromeDriverManager
except ImportError:
    print("Please install selenium and webdriver-manager (uv add selenium webdriver-manager)")
    sys.exit(1)

REVIEW_P = Path('data') / 'mappings' / 'player_map_review.csv'
SCRAPED_P = Path('data') / 'mappings' / 'scraped_attrs.csv'
FIFA1000_P = Path('data') / 'cache' / 'fifa_players_1000.parquet'
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
    # simply inspect all scripts and regex out the numbers.
    html = driver.page_source
    stats = {}
    import re
    for attr, var in JS_STATS.items():
        m = re.search(rf"{var}\s*=\s*(\d+)", html)
        stats[attr] = int(m.group(1)) if m else None
    return stats


def main():
    if not REVIEW_P.exists():
        print('review file not found', REVIEW_P)
        sys.exit(1)
    review = pd.read_csv(REVIEW_P, dtype={'player_id_sb': object, 'candidate_fifa_id': object, 'status': object})
    subset = review[review['status'].isin(['unmatched','review'])].copy()
    # only rows with a numeric candidate id
    subset['cand_id'] = pd.to_numeric(subset['candidate_fifa_id'], errors='coerce')
    subset = subset[pd.notna(subset['cand_id'])]
    print(f'Processing {len(subset)} rows with candidate fifa_id')
    if subset.empty:
        return

    # initialize webdriver once
    driver = init_driver()
    rows = []
    for idx, r in subset.iterrows():
        cand = int(r['cand_id'])
        url = f"https://sofifa.com/player/{cand}/{YEAR_ROSTER}"
        print('fetching', r['player_name_sb'], '->', url)
        try:
            driver.get(url)
            # sometimes Sofifa shows a redirect animation, wait a bit
            time.sleep(1)
            stats = extract_stats_from_page(driver)
            # debug snippet
            html = driver.page_source
            idx = html.find('POINT_PAC')
            if idx!=-1:
                snippet = html[idx:idx+200]
                print('  html snippet around POINT_PAC:', snippet)
            print('  stats:', stats)
            rows.append({
                'player_id_sb': r['player_id_sb'],
                'player_name_sb': r['player_name_sb'],
                'candidate_fifa_id': cand,
                **stats,
            })
        except Exception as e:
            print('error fetching', cand, e)
    driver.quit()

    if rows:
        df = pd.DataFrame(rows)
        if SCRAPED_P.exists():
            df.to_csv(SCRAPED_P, mode='a', header=False, index=False)
        else:
            df.to_csv(SCRAPED_P, index=False)
        print('Wrote scraped attributes to', SCRAPED_P)
    else:
        print('No rows scraped')

    # optional: copy attributes into players table for those rows
    try:
        players = pd.read_parquet(PLAYERS_P)
        for r in rows:
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
