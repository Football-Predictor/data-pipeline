#!/usr/bin/env python3
"""Fill missing player attribute values in players_train_1000.parquet from FIFA tables.

- For players with any NaN among ATTRS and with a fifa_id in player_map_1000.csv:
  * try to find row in fifa_players_1000.parquet by sofifa_id
  * if not present, try in global fifa_players.parquet and append to fifa_players_1000.parquet
  * copy pace/shooting/passing/dribbling/defending/physic where available

Writes updated players_train_1000.parquet and (optionally) appends to fifa_players_1000.parquet.
Prints summary of fixes and remaining missing players.
"""
# /// script
# dependencies = ["pandas","pyarrow"]
# ///
from pathlib import Path
import pandas as pd
import unicodedata

ROOT = Path('data')
PROC = ROOT / 'processed'
PLAYERS_P = PROC / 'players_train_1000.parquet'
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
    players = pd.read_parquet(PLAYERS_P)
    print('Loaded players_train_1000:', len(players))
    # load mapping subset
    if not PM1000.exists():
        print('Mapping subset not found:', PM1000)
        return
    pm = pd.read_csv(PM1000, dtype={'player_id_sb': int, 'fifa_id': object})

    # join mapping fifa_id onto players by player_name_sb
    merged = players.merge(pm[['player_name_sb','fifa_id']], on='player_name_sb', how='left')
    # normalize fifa_id strings
    merged['fifa_id_str'] = merged['fifa_id'].apply(lambda x: str(int(float(x))) if (pd.notna(x) and str(x).strip()!='nan') else None)

    # load fifa 1k subset
    fifa1000 = pd.read_parquet(FIFA1000)
    # ensure sofifa_id is string for matching
    if 'sofifa_id' in fifa1000.columns:
        fifa1000['sofifa_id_str'] = fifa1000['sofifa_id'].astype(str)
    else:
        fifa1000['sofifa_id_str'] = fifa1000.index.astype(str)

    # load global fifa
    fifa_full = None
    if FIFA_FULL.exists():
        fifa_full = pd.read_parquet(FIFA_FULL)
        if 'sofifa_id' in fifa_full.columns:
            fifa_full['sofifa_id_str'] = fifa_full['sofifa_id'].astype(str)
        else:
            fifa_full['sofifa_id_str'] = fifa_full.index.astype(str)

    fixed_from_subset = 0
    fixed_from_global = 0
    unresolved = []

    for i, r in merged.iterrows():
        has_missing = r[ATTRS].isna().any()
        if not has_missing:
            continue
        fid = r.get('fifa_id_str')
        if fid is None:
            unresolved.append((r['player_id'], r['player_name_sb'], 'no_fifa_id'))
            continue
        # try subset
        row_found = None
        m = fifa1000[fifa1000['sofifa_id_str']==fid]
        if len(m):
            row_found = m.iloc[0]
            # check if attributes present
            if any(pd.notna(row_found.get(a)) for a in ATTRS):
                # copy attributes into players dataframe
                for a in ATTRS:
                    if pd.isna(players.at[i,a]) and (a in row_found.index) and pd.notna(row_found.get(a)):
                        players.at[i,a] = row_found.get(a)
                fixed_from_subset += 1
                continue
        # not present or attributes missing in subset -> try global
        if fifa_full is not None:
            mg = fifa_full[fifa_full['sofifa_id_str']==fid]
            if len(mg):
                grow = mg.iloc[0]
                # append to subset if desired
                append_row = False
                # if subset missing this sofifa_id, append the full row to subset
                if len(m)==0:
                    append_row = True
                # if full row has attributes, copy
                if any(pd.notna(grow.get(a)) for a in ATTRS):
                    for a in ATTRS:
                        if pd.isna(players.at[i,a]) and (a in grow.index) and pd.notna(grow.get(a)):
                            players.at[i,a] = grow.get(a)
                    fixed_from_global += 1
                    # append full fifa row to subset so future runs see it
                    if append_row:
                        fifa1000 = pd.concat([fifa1000, pd.DataFrame([grow])], ignore_index=True)
                        print('Appended fifa row to fifa_players_1000.parquet for sofifa_id', fid)
                    continue
        # if reached here, couldn't fix
        unresolved.append((r['player_id'], r['player_name_sb'], fid))

    # write back updated players and fifa subset if modified
    players.to_parquet(PLAYERS_P, index=False)
    print('Wrote updated players_train_1000.parquet')
    if fixed_from_global>0:
        fifa1000.to_parquet(FIFA1000, index=False)
        print('Wrote updated fifa_players_1000.parquet (appended rows from global)')

    print('\nSummary fixes:')
    print(' - fixed_from_subset:', fixed_from_subset)
    print(' - fixed_from_global:', fixed_from_global)
    print(' - unresolved_count:', len(unresolved))
    if unresolved:
        print('\nSample unresolved entries (player_id, name, fifa_id):')
        for ex in unresolved[:20]:
            print(' ', ex)

if __name__ == '__main__':
    main()
