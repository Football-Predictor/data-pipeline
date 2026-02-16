#!/usr/bin/env python3
"""Integrity checks for players/matches training tables.
"""
# /// script
# dependencies = ["pandas","pyarrow"]
# ///
import pandas as pd
from pathlib import Path

P_PLAY = Path('data/processed/players_train_1000.parquet')
P_MATCH = Path('data/processed/matches_train_1000.parquet')
P_SP = Path('data/cache/matches_starting_players.parquet')

players = pd.read_parquet(P_PLAY)
matches = pd.read_parquet(P_MATCH)
sp = pd.read_parquet(P_SP)

print('players_train rows:', len(players))
# fifa_id nulls
null_fifa = players['fifa_id'].isna().sum()
print('players with null fifa_id:', null_fifa, f'({null_fifa/len(players):.2%})')
# players with null fifa_id but with at least one attribute present
attrs = ['pace','shooting','passing','dribbling','defending','physic']
has_attr_but_no_id = players[players['fifa_id'].isna() & players[attrs].notna().any(axis=1)]
print('players with NULL fifa_id but at least one attribute present:', len(has_attr_but_no_id))
print('\nSample rows (NULL fifa_id but attributes present):')
print(has_attr_but_no_id.head(10).to_string(index=False))

# check for fifa_id == 0 or '0'
zero_ids = players['fifa_id'].apply(lambda x: str(x).strip()=='0').sum()
print('\nplayers with fifa_id == 0:', zero_ids)

# matches: inspect starting xi lists for 0 or missing entries
def _as_list_like(x):
    # normalize different container types to a Python list
    import ast
    import numpy as np
    if isinstance(x, (list, tuple)):
        return list(x)
    if hasattr(x, 'tolist') and not isinstance(x, str):
        try:
            return list(x.tolist())
        except Exception:
            pass
    if isinstance(x, (str,)):
        # try to parse stringified list
        try:
            v = ast.literal_eval(x)
            if isinstance(v, (list, tuple)):
                return list(v)
        except Exception:
            return []
    return []


def flatten_lists(col):
    vals = []
    for lst in matches[col].fillna(pd.NA):
        items = _as_list_like(lst)
        vals.extend(items)
    return vals

home_vals = flatten_lists('home_starting_xi')
away_vals = flatten_lists('away_starting_xi')
all_vals = [v for v in (home_vals+away_vals) if pd.notna(v)]
# count '0' occurrences and invalid ids (0 or empty strings)
zero_in_matches = sum(1 for v in all_vals if str(v).strip()=='0')
nan_in_matches = sum(1 for v in all_vals if pd.isna(v))
print('\nTotal starting-xi entries scanned (home+away):', len(all_vals))
print('Entries equal to 0:', zero_in_matches)
print('Entries that are NaN:', nan_in_matches)

# show example matches containing 0 or NaN entries
bad_matches = []
for _, r in matches.iterrows():
    bad = False
    for side in ['home_starting_xi','away_starting_xi']:
        lst = _as_list_like(r.get(side))
        if any(str(x).strip()=='0' for x in lst) or any(pd.isna(x) for x in lst):
            bad = True
    if bad:
        bad_matches.append(r['match_id'])
print('\nMatches with missing/0 player ids (first 10):', bad_matches[:10])
print('Count of affected matches:', len(set(bad_matches)))

# inspect source starting_players parquet for player_id_sb==0 or missing
sp_zero = sp[ (sp['player_id_sb']==0) | (sp['player_id_sb'].isna()) ]
print('\nRows in matches_starting_players.parquet with player_id_sb == 0 or NaN:', len(sp_zero))
print(sp_zero.head(10).to_string(index=False))

# Provide guidance summary
print('\nGuidance:')
print(' - To fix fifa_id nulls for players that have attribute values: re-run mapping for those player_name_sb using a name-based lookup against FIFA rows (we can attempt exact + token-blocked fuzzy).')
print(' - To fix 0 player ids in matches: replace 0 with NaN in `matches_train_1000.parquet` and, if present upstream, in `data/cache/matches_starting_players.parquet` (source should not have player_id_sb==0).')
print(' - Optionally remove match rows with too many missing players or mark them as unusable for training.')
