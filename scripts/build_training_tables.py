#!/usr/bin/env python3
"""Build two training tables:
 - players_train.parquet: one row per StatsBomb player appearing in first 1000 matches, joined to FIFA attributes
 - matches_train.parquet: one row per match (first 1000) with starting XI lists (fifa_id list), home/away, and match labels

Outputs:
 - data/processed/players_train_1000.parquet
 - data/processed/matches_train_1000.parquet

Behavior: uses existing `data/cache/fifa_players_1000.parquet`, `data/mappings/player_map_1000.csv`, and `data/cache/matches_starting_players.parquet`.
"""
# /// script
# dependencies = ["pandas","pyarrow","rapidfuzz"]
# ///
from pathlib import Path
import pandas as pd
import unicodedata
from rapidfuzz import fuzz

ROOT = Path('data')
OUT = ROOT / 'processed'
OUT.mkdir(parents=True, exist_ok=True)

SP_P = ROOT / 'cache' / 'matches_starting_players.parquet'
MATCHES_P = ROOT / 'cache' / 'matches.parquet'
FIFA_P = ROOT / 'cache' / 'fifa_players_1000.parquet'
PLAYER_MAP = ROOT / 'mappings' / 'player_map_1000.csv'

N = 1000

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


def choose_best_fifa_row(name_sb_norm, candidates_df):
    # choose best candidate by token_set_ratio on normalized names
    best_idx = None
    best_score = -1
    for i, r in candidates_df.iterrows():
        cand = normalize_name(str(r.get('short_name','')) + ' || ' + str(r.get('long_name','')))
        s = fuzz.token_set_ratio(name_sb_norm, cand)
        if s > best_score:
            best_score = s
            best_idx = i
    return best_idx, best_score


def main():
    sp = pd.read_parquet(SP_P)
    match_ids = sp['match_id'].dropna().astype(int).unique().tolist()
    first_n = set(match_ids[:N])
    sp_n = sp[sp['match_id'].isin(first_n)].copy()

    # load player map for the 1k subset
    pm = pd.read_csv(PLAYER_MAP, dtype={'player_id_sb': int, 'fifa_id': object})
    pm_map = {int(r['player_id_sb']): r for _, r in pm.iterrows()}

    # load fifa players (reduced parquet)
    fifa = pd.read_parquet(FIFA_P)
    # add a normalized name column for lookup
    fifa['_norm_short'] = fifa.get('short_name','').fillna('').astype(str).apply(normalize_name)
    fifa['_norm_long'] = fifa.get('long_name','').fillna('').astype(str).apply(normalize_name)
    fifa['_norm_combined'] = (fifa['_norm_short'] + ' || ' + fifa['_norm_long']).fillna('')

    # build a token -> indices index for fast blocking
    token_index = {}
    for idx, n in fifa['_norm_combined'].items():
        for t in set([tok for tok in n.split() if len(tok) > 1]):
            token_index.setdefault(t, []).append(idx)

    MAX_CANDIDS = 5000

    # build player table: one row per unique StatsBomb player in first N matches
    sb_players = sp_n[['player_id_sb','player_name_sb']].drop_duplicates().copy()
    rows = []
    for _, r in sb_players.iterrows():
        sbid = int(r['player_id_sb'])
        sbname = r['player_name_sb']
        sbn = normalize_name(sbname)
        fifa_row = None
        fifa_id = None
        mapped = pm_map.get(sbid)
        # 1) if mapping provides a fifa_id and fifa has sofifa_id column, match on it
        if mapped is not None:
            mapped_fifa_id = str(mapped.get('fifa_id')) if pd.notna(mapped.get('fifa_id')) else None
            if 'sofifa_id' in fifa.columns and mapped_fifa_id is not None:
                match_df = fifa[fifa['sofifa_id'].astype(str)==mapped_fifa_id]
                if len(match_df):
                    fifa_row = match_df.iloc[0]
                    fifa_id = mapped_fifa_id
        # 2) exact normalized short/long match
        if fifa_row is None:
            cand_idx = fifa.index[(fifa['_norm_short']==sbn) | (fifa['_norm_long']==sbn)].tolist()
            if len(cand_idx)==1:
                fifa_row = fifa.loc[cand_idx[0]]
            elif len(cand_idx) > 1:
                # choose best among candidates
                choices = fifa.loc[cand_idx]['_norm_combined'].tolist()
                from rapidfuzz import process
                res = process.extractOne(sbn, choices, scorer=fuzz.token_set_ratio)
                if res:
                    _, score, local_i = res
                    global_idx = cand_idx[local_i]
                    fifa_row = fifa.loc[global_idx]
        # 3) token-blocked fuzzy search (fast)
        if fifa_row is None:
            toks = [t for t in sbn.split() if len(t)>1]
            cand_sets = [set(token_index.get(t, [])) for t in toks]
            if cand_sets:
                cand_union = set().union(*cand_sets)
            else:
                cand_union = set()
            if cand_union and len(cand_union) <= MAX_CANDIDS:
                cand_idx = list(cand_union)
                choices = fifa.loc[cand_idx]['_norm_combined'].tolist()
                from rapidfuzz import process
                res = process.extractOne(sbn, choices, scorer=fuzz.token_set_ratio)
                if res:
                    _, score, local_i = res
                    global_idx = cand_idx[local_i]
                    fifa_row = fifa.loc[global_idx]
        # 4) if still None, leave fifa_row as None

        # collect attributes (keep NaNs as requested)
        attr_vals = {a: (fifa_row.get(a) if (fifa_row is not None and a in fifa_row.index) else pd.NA) for a in ATTRS}
        out = {
            'player_id_sb': sbid,
            'player_name_sb': sbname,
            'fifa_id': (fifa_row.get('sofifa_id') if (fifa_row is not None and 'sofifa_id' in fifa_row.index)
                        else (mapped.get('fifa_id') if mapped is not None else pd.NA)),
            'fifa_short_name': (fifa_row.get('short_name') if fifa_row is not None else pd.NA),
        }
        out.update(attr_vals)
        rows.append(out)

    players_df = pd.DataFrame(rows)
    players_out = OUT / 'players_train_1000.parquet'
    players_df.to_parquet(players_out, index=False)
    print('Wrote player table:', players_out, 'rows=', len(players_df))

    # Build match table: aggregate starting XI into lists of fifa_id (or NaN if not found)
    # need match-level info (scores, home/away team ids)
    matches = pd.read_parquet(MATCHES_P)
    matches_n = matches[matches['match_id'].isin(first_n)].copy()
    # ensure required score columns
    home_score_col = None
    away_score_col = None
    for cand in ['home_score','home_goals','home_team_score','home_team_goals']:
        if cand in matches_n.columns:
            home_score_col = cand; break
    for cand in ['away_score','away_goals','away_team_score','away_team_goals']:
        if cand in matches_n.columns:
            away_score_col = cand; break
    if home_score_col is None or away_score_col is None:
        print('Warning: could not find home/away score columns; match labels will be empty')

    # build mapping player_id_sb -> fifa_id using players_df
    pid_to_fifa = dict(zip(players_df['player_id_sb'], players_df['fifa_id']))

    match_rows = []
    for mid, grp in sp_n.groupby('match_id'):
        # determine home/away team ids from matches_n
        mrow = matches_n[matches_n['match_id']==mid]
        if len(mrow):
            mrow = mrow.iloc[0]
            home_team = mrow.get('home_team_id') if 'home_team_id' in mrow.index else mrow.get('home_team') if 'home_team' in mrow.index else pd.NA
            away_team = mrow.get('away_team_id') if 'away_team_id' in mrow.index else mrow.get('away_team') if 'away_team' in mrow.index else pd.NA
            home_goals = mrow.get(home_score_col) if home_score_col else pd.NA
            away_goals = mrow.get(away_score_col) if away_score_col else pd.NA
        else:
            home_team = pd.NA; away_team = pd.NA; home_goals = pd.NA; away_goals = pd.NA
        # group rows by team_id and build ordered player lists by jersey
        teams = {}
        for _, pr in grp.iterrows():
            tid = pr['team_id']
            teams.setdefault(tid, []).append(pr)
        # determine which team is home/away by matching team ids if possible
        home_players = []
        away_players = []
        for tid, players in teams.items():
            # sort by jersey
            players_sorted = sorted(players, key=lambda x: (pd.isnull(x.get('jersey')), x.get('jersey')))
            fifa_ids = [pid_to_fifa.get(int(p['player_id_sb']), pd.NA) for p in players_sorted]
            if not pd.isna(home_team) and tid == home_team:
                home_players = fifa_ids
            elif not pd.isna(away_team) and tid == away_team:
                away_players = fifa_ids
            else:
                # if home/away not available, assign first team to home if empty
                if not home_players:
                    home_players = fifa_ids
                else:
                    away_players = fifa_ids
        # ensure lists length <=11
        home_players = home_players[:11]
        away_players = away_players[:11]
        # compute result
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

    matches_out_df = pd.DataFrame(match_rows)
    matches_out = OUT / 'matches_train_1000.parquet'
    matches_out_df.to_parquet(matches_out, index=False)
    print('Wrote match table:', matches_out, 'rows=', len(matches_out_df))

    print('\nDone. Output files in', OUT)

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise
