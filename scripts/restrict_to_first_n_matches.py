#!/usr/bin/env python3
"""Restrict workspace to players appearing in the first N matches.

Outputs:
- data/mappings/player_map_review.csv (overwritten, backup kept .bak)
- data/mappings/player_map_review_1000.csv (subset)
- data/mappings/player_map_1000.csv (accepted mappings subset)
- data/cache/fifa_players_1000.parquet (FIFA rows matching players in first N matches by exact normalized name)

Usage: uv run scripts/restrict_to_first_n_matches.py
"""
# /// script
# dependencies = ["pandas","pyarrow"]
# ///
from pathlib import Path
import pandas as pd
import unicodedata

ROOT = Path('data')
SP_P = ROOT / 'cache' / 'matches_starting_players.parquet'
REVIEW_P = ROOT / 'mappings' / 'player_map_review.csv'
REVIEW_OUT = ROOT / 'mappings' / 'player_map_review_1000.csv'
REVIEW_BAK = ROOT / 'mappings' / 'player_map_review.csv.bak'
ACCEPT_P = ROOT / 'mappings' / 'player_map.csv'
ACCEPT_OUT = ROOT / 'mappings' / 'player_map_1000.csv'
FIFA_P = ROOT / 'cache' / 'fifa_players.parquet'
FIFA_OUT = ROOT / 'cache' / 'fifa_players_1000.parquet'

N = 1000


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
    sp = pd.read_parquet(SP_P)
    ordered_match_ids = sp['match_id'].dropna().astype(int).unique()
    first_n = set(ordered_match_ids[:N])
    print('Selected first', N, 'match_ids ->', len(first_n))

    sp_n = sp[sp['match_id'].isin(first_n)].copy()
    players_in_n = set(sp_n['player_id_sb'].dropna().astype(int).unique())
    print('Unique StatsBomb player ids in first', N, 'matches:', len(players_in_n))

    # filter review file
    review = pd.read_csv(REVIEW_P, dtype={'player_id_sb': int})
    review_filtered = review[review['player_id_sb'].isin(players_in_n)].copy()
    # backup original review file
    REVIEW_BAK.parent.mkdir(parents=True, exist_ok=True)
    REVIEW_BAK.write_bytes(Path(REVIEW_P).read_bytes())
    # write filtered file and a separate copy
    review_filtered.to_csv(REVIEW_P, index=False)
    review_filtered.to_csv(REVIEW_OUT, index=False)
    print('Wrote filtered review file:', REVIEW_P, 'rows kept:', len(review_filtered))

    # filter accepted mapping file
    accepted = pd.read_csv(ACCEPT_P, dtype={'player_id_sb': int})
    accepted_filtered = accepted[accepted['player_id_sb'].isin(players_in_n)].copy()
    accepted_filtered.to_csv(ACCEPT_OUT, index=False)
    print('Wrote accepted mapping subset:', ACCEPT_OUT, 'rows kept:', len(accepted_filtered))

    # build set of normalized player names from both starting players and accepted mapping
    names_sb = set(normalize_name(n) for n in sp_n['player_name_sb'].dropna().unique().tolist())
    names_map = set(normalize_name(n) for n in accepted_filtered['player_name_sb'].dropna().unique().tolist())
    target_names = names_sb.union(names_map)
    print('Unique normalized player names to keep (approx):', len(target_names))

    # filter fifa parquet by exact normalized short_name or long_name
    fifa = pd.read_parquet(FIFA_P)
    # build normalized columns
    fifa['_norm_short'] = fifa.get('short_name','').fillna('').astype(str).apply(normalize_name)
    fifa['_norm_long'] = fifa.get('long_name','').fillna('').astype(str).apply(normalize_name)
    mask = fifa['_norm_short'].isin(target_names) | fifa['_norm_long'].isin(target_names)
    fifa_kept = fifa[mask].copy()
    # drop helper cols
    fifa_kept = fifa_kept.drop(columns=['_norm_short','_norm_long'], errors='ignore')
    fifa_kept.to_parquet(FIFA_OUT, index=False)
    print('Wrote filtered fifa players to', FIFA_OUT, 'rows kept:', len(fifa_kept), 'of', len(fifa))

    # report counts
    print('\nSummary:')
    print(' - review rows: original ->', len(review), ', filtered ->', len(review_filtered))
    print(' - accepted mapping rows: original ->', len(accepted), ', filtered ->', len(accepted_filtered))
    print(' - fifa players: original ->', len(fifa), ', filtered ->', len(fifa_kept))

if __name__=='__main__':
    main()
