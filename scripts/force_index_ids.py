#!/usr/bin/env python3
"""Force player IDs in processed training tables to be index+1 and update matches.

- Overwrite `data/processed/players_train_1000.parquet` so `fifa_id = index + 1` (int)
- Rebuild `data/processed/matches_train_1000.parquet` so starting XI lists reference the new ids
- Update `data/mappings/player_map_1000.csv` to reflect new fifa_id for players in the subset
- Update `data/cache/fifa_players_1000.parquet` to set `sofifa_id = index + 1` for consistency
- Back up overwritten files (`*.bak`)

This makes the processed dataset self-contained and ensures the web app/model can use numeric ids consistently.
"""
# /// script
# dependencies = ["pandas","pyarrow"]
# ///
from pathlib import Path
import pandas as pd
import shutil

ROOT = Path('data')
PROC = ROOT / 'processed'
PLAYERS_P = PROC / 'players_train_1000.parquet'
MATCHES_P = PROC / 'matches_train_1000.parquet'
PM1000 = ROOT / 'mappings' / 'player_map_1000.csv'
FIFA_1000 = ROOT / 'cache' / 'fifa_players_1000.parquet'
SP = ROOT / 'cache' / 'matches_starting_players.parquet'

BACKUP_SUFFIX = '.bak'

# helpers
def backup_if_exists(p: Path):
    if p.exists():
        bak = p.with_suffix(p.suffix + BACKUP_SUFFIX)
        shutil.copy2(p, bak)
        print('Backed up', p, '->', bak)


def main():
    # load players_train
    players = pd.read_parquet(PLAYERS_P)
    print('Loaded players_train rows=', len(players))
    backup_if_exists(PLAYERS_P)

    # assign new id = index + 1
    players = players.reset_index(drop=True)
    players['fifa_id'] = (players.index + 1).astype(int)
    # persist change
    players.to_parquet(PLAYERS_P, index=False)
    print('Wrote updated', PLAYERS_P)

    # build pid_sb -> new_id map
    pid_to_new = dict(zip(players['player_id_sb'].astype(int), players['fifa_id'].astype(int)))

    # update matches_train by reconstructing from matches_starting_players.parquet for first 1000 matches
    sp = pd.read_parquet(SP)
    match_ids = sp['match_id'].dropna().astype(int).unique().tolist()
    first_n = set(match_ids[:1000])
    sp_n = sp[sp['match_id'].isin(first_n)].copy()

    # load original matches (scores/team ids)
    matches = pd.read_parquet(ROOT / 'cache' / 'matches.parquet')
    matches_n = matches[matches['match_id'].isin(first_n)].copy()

    match_rows = []
    for mid, grp in sp_n.groupby('match_id'):
        mrow = matches_n[matches_n['match_id']==mid]
        if len(mrow):
            mrow = mrow.iloc[0]
            home_team = mrow.get('home_team_id') if 'home_team_id' in mrow.index else mrow.get('home_team') if 'home_team' in mrow.index else pd.NA
            away_team = mrow.get('away_team_id') if 'away_team_id' in mrow.index else mrow.get('away_team') if 'away_team' in mrow.index else pd.NA
            home_goals = mrow.get('home_score') if 'home_score' in mrow.index else (mrow.get('home_goals') if 'home_goals' in mrow.index else pd.NA)
            away_goals = mrow.get('away_score') if 'away_score' in mrow.index else (mrow.get('away_goals') if 'away_goals' in mrow.index else pd.NA)
        else:
            home_team = pd.NA; away_team = pd.NA; home_goals = pd.NA; away_goals = pd.NA
        teams = {}
        for _, pr in grp.iterrows():
            tid = pr['team_id']
            teams.setdefault(tid, []).append(pr)
        home_players = []
        away_players = []
        for tid, players_list in teams.items():
            players_sorted = sorted(players_list, key=lambda x: (pd.isnull(x.get('jersey')), x.get('jersey')))
            fifa_ids = []
            for p in players_sorted:
                pid = int(p['player_id_sb']) if pd.notna(p['player_id_sb']) else None
                if pid is None:
                    fifa_ids.append(pd.NA)
                else:
                    newid = pid_to_new.get(pid)
                    fifa_ids.append(newid if newid is not None else pd.NA)
            if 'home_team_id' in mrow.index and tid == home_team:
                home_players = fifa_ids
            elif 'away_team_id' in mrow.index and tid == away_team:
                away_players = fifa_ids
            else:
                if not home_players:
                    home_players = fifa_ids
                else:
                    away_players = fifa_ids
        home_players = home_players[:11]
        away_players = away_players[:11]
        if pd.isna(home_goals) or pd.isna(away_goals):
            result = pd.NA
        else:
            if home_goals > away_goals:
                result = 'H'
            elif home_goals < away_goals:
                result = 'A'
            else:
                result = 'D'
        match_rows.append({
            'match_id': int(mid),
            'home_team_id': home_team,
            'away_team_id': away_team,
            'home_starting_xi': home_players,
            'away_starting_xi': away_players,
            'home_goals': home_goals,
            'away_goals': away_goals,
            'result': result,
        })

    match_df = pd.DataFrame(match_rows)
    backup_if_exists(MATCHES_P)
    match_df.to_parquet(MATCHES_P, index=False)
    print('Wrote updated', MATCHES_P, 'rows=', len(match_df))

    # update player_map_1000.csv fifa_id values for subset
    if PM1000.exists():
        pm = pd.read_csv(PM1000, dtype={'player_id_sb': int})
        updated = 0
        for i, r in pm.iterrows():
            pid = int(r['player_id_sb'])
            if pid in pid_to_new:
                newid = pid_to_new[pid]
                if str(r.get('fifa_id')) != str(newid):
                    pm.at[i,'fifa_id'] = newid
                    updated += 1
        if updated:
            shutil.copy2(PM1000, PM1000.with_suffix(PM1000.suffix + BACKUP_SUFFIX))
            pm.to_csv(PM1000, index=False)
            print('Updated', updated, 'rows in', PM1000)
        else:
            print('No updates needed in', PM1000)

    # update fifa_players_1000.parquet sofifa_id to index+1 for consistency
    if FIFA_1000.exists():
        fifa = pd.read_parquet(FIFA_1000)
        fifa = fifa.reset_index(drop=True)
        fifa['sofifa_id'] = (fifa.index + 1).astype(int)
        backup_if_exists(FIFA_1000)
        fifa.to_parquet(FIFA_1000, index=False)
        print('Updated sofifa_id in', FIFA_1000)

    # run integrity check
    print('\nRe-running integrity checks:')
    import subprocess
    subprocess.check_call(['uv','run','scripts/check_player_match_integrity.py'])

if __name__=='__main__':
    main()
