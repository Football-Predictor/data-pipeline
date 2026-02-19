#!/usr/bin/env python3
"""Diagnose players in players_train_1000.parquet with missing FIFA attribute values.

Checks:
 - how many players have any NaN among [pace,shooting,passing,dribbling,defending,physic]
 - whether those players have entries in player_map_1000.csv and if a fifa_id is recorded
 - whether those fifa_id values exist in fifa_players_1000.parquet (sofifa_id or normalized name)

Run: uv run scripts/diagnose_missing_player_attrs.py
"""
# /// script
# dependencies = ["pandas","pyarrow"]
# ///
from pathlib import Path
import pandas as pd
import unicodedata

ROOT = Path('data')
PROC = ROOT / 'processed'
PLAY = PROC / 'players_train_1000.parquet'
PM1000 = ROOT / 'mappings' / 'player_map_1000.csv'
FIFA1000 = ROOT / 'cache' / 'fifa_players_1000.parquet'
FIFA_FULL = ROOT / 'cache' / 'fifa_players.parquet'

ATTRS = ['pace','shooting','passing','dribbling','defending','physic']


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
    p = pd.read_parquet(PLAY)
    print('Loaded processed players rows=', len(p))
    # rows with any NaN among ATTRS
    mask_anyna = p[ATTRS].isna().any(axis=1)
    missing_players = p[mask_anyna].copy()
    total_missing = len(missing_players)
    pct = total_missing / len(p) * 100
    print(f'Players with ANY missing attribute among {ATTRS}: {total_missing} ({pct:.2f}%)')
    if total_missing==0:
        return
    print('\nSample players with missing attributes:')
    print(missing_players.head(12).to_string(index=False))

    # load mapping subset
    pm = pd.read_csv(PM1000, dtype={'player_id_sb': int, 'fifa_id': object}) if PM1000.exists() else pd.DataFrame()
    # normalize names in FIFA 1000
    fifa = pd.read_parquet(FIFA1000)
    fifa['_norm_short'] = fifa.get('short_name','').fillna('').astype(str).apply(normalize_name)
    fifa['_norm_long'] = fifa.get('long_name','').fillna('').astype(str).apply(normalize_name)
    sofifa_ids = set()
    if 'sofifa_id' in fifa.columns:
        sofifa_ids = set(fifa['sofifa_id'].astype(str).tolist())

    # analyze mapping existence for missing players
    mapped_count = 0
    mapped_but_no_fifa_row = 0
    unmapped_count = 0
    examples = []
    for _, r in missing_players.iterrows():
        name = r['player_name_sb']
        # try find mapping by name in pm
        match = pm[pm['player_name_sb'] == name]
        if len(match):
            mapped_count += 1
            fid = match.iloc[0].get('fifa_id')
            if pd.isna(fid) or str(fid).strip()=='' or str(fid).strip()=='nan':
                unmapped_count += 1
                examples.append((r['player_id'], name, None, 'mapped_no_fifa_id'))
            else:
                # check fifa row existence
                fid_s = str(int(float(fid))) if (isinstance(fid, (int,float)) and not pd.isna(fid)) else str(fid)
                if fid_s in sofifa_ids:
                    # fifa row exists but attributes still missing => odd
                    examples.append((r['player_id'], name, fid_s, 'mapped_and_fifa_row_exists'))
                else:
                    mapped_but_no_fifa_row += 1
                    examples.append((r['player_id'], name, fid_s, 'mapped_but_no_fifa_row'))
        else:
            unmapped_count += 1
            examples.append((r['player_id'], name, None, 'no_mapping'))
    print(f'\nMapping status for players with missing attributes:')
    print(' - mapped (has mapping row):', mapped_count)
    print(' - mapped but fifa row missing from fifa_players_1000.parquet:', mapped_but_no_fifa_row)
    print(' - unmapped (no fifa_id in mapping subset):', unmapped_count)
    print('\nExamples (player_id, name, mapped_fifa_id, status):')
    for ex in examples[:20]:
        print(' ', ex)

    # recommendation summary
    print('\nRecommendations:')
    print('- If many rows are "mapped_but_no_fifa_row": rebuild fifa_players_1000.parquet or expand fifa subset so mapped sofifa_id rows are included.')
    print('- If many rows are "no_mapping" or "mapped_no_fifa_id": re-run targeted fuzzy/surname/initial passes or use review CLI to add mappings.')
    print('- For training, consider imputing attributes (median by position) or flagging players so model handles missing values explicitly.')

if __name__=='__main__':
    main()
