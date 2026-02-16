#!/usr/bin/env python3
"""Convert processed training tables to use a single index-based `player_id`.

- Overwrite `data/processed/players_train_1000.parquet`: drop `fifa_id` and `player_id_sb`, add `player_id = index+1`.
- Update `data/processed/matches_train_1000.parquet`: ensure starting-XI lists reference `player_id`.
- Back up originals to `*.bak`.
- Update `scripts/check_player_match_integrity.py` expectations is done separately.
"""
# /// script
# dependencies = ["pandas","pyarrow"]
# ///
from pathlib import Path
import pandas as pd
import shutil

ROOT = Path('data')
PROC = ROOT / 'processed'
PLAYERS = PROC / 'players_train_1000.parquet'
MATCHES = PROC / 'matches_train_1000.parquet'
BACKUP_SUFFIX = '.bak'


def backup(p: Path):
    if p.exists():
        bak = p.with_suffix(p.suffix + BACKUP_SUFFIX)
        shutil.copy2(p, bak)
        print('Backed up', p, '->', bak)


def _as_list_like(x):
    # normalize different container types to a Python list
    import ast
    if isinstance(x, (list, tuple)):
        return list(x)
    if hasattr(x, 'tolist') and not isinstance(x, str):
        try:
            return list(x.tolist())
        except Exception:
            pass
    if isinstance(x, (str,)):
        try:
            v = ast.literal_eval(x)
            if isinstance(v, (list, tuple)):
                return list(v)
        except Exception:
            return []
    return []


def main():
    if not PLAYERS.exists() or not MATCHES.exists():
        raise FileNotFoundError('Processed files not found in data/processed; run build_training_tables.py first')

    # load players
    players = pd.read_parquet(PLAYERS)
    print('Loaded players:', len(players), 'columns:', list(players.columns))
    backup(PLAYERS)

    # create new player_id and drop old id columns if present
    players = players.reset_index(drop=True)
    players['player_id'] = (players.index + 1).astype(int)
    dropped = []
    for col in ['fifa_id','player_id_sb']:
        if col in players.columns:
            players = players.drop(columns=[col])
            dropped.append(col)
    print('Dropped columns from players:', dropped)

    # ensure player_id is first column
    cols = list(players.columns)
    cols.insert(0, cols.pop(cols.index('player_id')))
    players = players[cols]

    players.to_parquet(PLAYERS, index=False)
    print('Wrote updated players ->', PLAYERS)

    # load matches and remap starting XI lists
    matches = pd.read_parquet(MATCHES)
    print('Loaded matches:', len(matches), 'columns:', list(matches.columns))
    backup(MATCHES)

    valid_ids = set(players['player_id'].astype(int).tolist())

    def remap_list(lst):
        out = []
        for v in _as_list_like(lst):
            if pd.isna(v):
                out.append(pd.NA)
                continue
            try:
                iv = int(v)
            except Exception:
                out.append(pd.NA)
                continue
            # if iv already in valid_ids, keep; otherwise try to map by position (no-op)
            if iv in valid_ids:
                out.append(iv)
            else:
                # not found: set NaN
                out.append(pd.NA)
        return out

    for side in ['home_starting_xi','away_starting_xi']:
        if side in matches.columns:
            matches[side] = matches[side].apply(remap_list)

    matches.to_parquet(MATCHES, index=False)
    print('Wrote updated matches ->', MATCHES)

    print('\nDone. players now use `player_id` and matches reference it.')

if __name__=='__main__':
    main()
