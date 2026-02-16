#!/usr/bin/env python3
"""Assign sofifa_id = row_index + 1 for fifa_players_1000.parquet and fill missing mappings.

Then re-run the training-table build and integrity checks.
"""
# /// script
# dependencies = ["pandas","pyarrow"]
# ///
from pathlib import Path
import pandas as pd
import unicodedata

ROOT = Path('data')
FIFA_P = ROOT / 'cache' / 'fifa_players_1000.parquet'
FIFA_FULL = ROOT / 'cache' / 'fifa_players.parquet'
PM1000 = ROOT / 'mappings' / 'player_map_1000.csv'
PM = ROOT / 'mappings' / 'player_map.csv'


def normalize_name(s: str) -> str:
    if not isinstance(s, str):
        return ''
    s = s.lower()
    s = unicodedata.normalize('NFKD', s)
    s = ''.join([c for c in s if not unicodedata.combining(c)])
    s = s.replace("'", "").replace('"','').replace('.','')
    s = ' '.join(s.split())
    return s


def main():
    # load reduced FIFA parquet
    fifa = pd.read_parquet(FIFA_P)
    print('Loaded fifa_players_1000.parquet rows=', len(fifa))

    # create sofifa_id where missing using index+1
    if 'sofifa_id' not in fifa.columns or fifa['sofifa_id'].isna().any():
        fifa = fifa.reset_index(drop=True)
        fifa['sofifa_id'] = (fifa.index + 1).astype(int)
        print('Assigned sofifa_id = index+1 for fifa_players_1000.parquet')
    else:
        print('sofifa_id column exists and appears populated')

    # write back
    fifa.to_parquet(FIFA_P, index=False)
    print('Wrote updated', FIFA_P)

    # build normalized name columns for matching
    fifa['_norm_short'] = fifa.get('short_name','').fillna('').astype(str).apply(normalize_name)
    fifa['_norm_long'] = fifa.get('long_name','').fillna('').astype(str).apply(normalize_name)
    norm_map = {n: sof for n, sof in zip(fifa['_norm_short'].tolist(), fifa['sofifa_id'].tolist())}
    # also include long names
    norm_map.update({n: sof for n, sof in zip(fifa['_norm_long'].tolist(), fifa['sofifa_id'].tolist())})

    # fill missing fifa_id in mapping subset
    pm = pd.read_csv(PM1000, dtype={'player_id_sb': int, 'fifa_id': object})
    changed = 0
    for i, r in pm.iterrows():
        fid = r.get('fifa_id')
        if pd.isna(fid) or str(fid).strip()=='' or str(fid).strip()=='0':
            n = normalize_name(r['player_name_sb'])
            if n in norm_map:
                pm.at[i,'fifa_id'] = int(norm_map[n])
                pm.at[i,'method'] = (r.get('method') or '') + '|synth_index'
                changed += 1
    if changed:
        pm.to_csv(PM1000, index=False)
        print(f'Filled {changed} missing fifa_id values in {PM1000}')
    else:
        print('No missing fifa_id values filled in mapping subset')

    # Also update global player_map.csv for any rows where fifa_id is missing and player present in pm1000
    if PM.exists():
        pm_global = pd.read_csv(PM, dtype={'player_id_sb': int, 'fifa_id': object})
        merged = pm_global.merge(pm[['player_id_sb','fifa_id']], on='player_id_sb', how='left', suffixes=('','_1000'))
        updates = 0
        for i, r in merged.iterrows():
            if (pd.isna(r['fifa_id']) or str(r['fifa_id']).strip()=='') and pd.notna(r.get('fifa_id_1000')):
                pm_global.at[i,'fifa_id'] = r['fifa_id_1000']
                pm_global.at[i,'method'] = (r.get('method') or '') + '|synth_index'
                updates += 1
        if updates:
            pm_global.to_csv(PM, index=False)
            print(f'Updated {updates} rows in global {PM} from {PM1000}')
        else:
            print('No updates to global player_map.csv needed')
    else:
        print('Global player_map.csv not found; skipping global update')

    # Rebuild training tables so players/matches receive the new sofifa_id values
    print('Rebuilding processed training tables...')
    import subprocess
    subprocess.check_call(['uv','run','scripts/build_training_tables.py'])

    # run integrity check
    subprocess.check_call(['uv','run','scripts/check_player_match_integrity.py'])

if __name__ == '__main__':
    main()
