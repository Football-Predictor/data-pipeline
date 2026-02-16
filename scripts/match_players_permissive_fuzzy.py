#!/usr/bin/env python3
"""Permissive fuzzy pass targeted to currently-unmatched players.

- Lowers fuzzy score cutoff to find candidates for unmatched SB players
- Accepts matches with score >= 75 (accepted_permissive)
- Accepts score >=70 when position group matches (accepted_permissive_pos)
- Marks others with candidate info and status='review' when score >=70

Usage: uv run scripts/match_players_permissive_fuzzy.py
"""
# /// script
# dependencies = ["pandas", "pyarrow", "rapidfuzz"]
# ///
from pathlib import Path
import pandas as pd
from rapidfuzz import process, fuzz
import unicodedata

ROOT = Path('data')
MAPDIR = ROOT / 'mappings'
REVIEW_P = MAPDIR / 'player_map_review.csv'
ACCEPT_P = MAPDIR / 'player_map.csv'
FIFA_PARQ = ROOT / 'cache' / 'fifa_players.parquet'
SP_P = ROOT / 'cache' / 'matches_starting_players.parquet'

# thresholds & caps
PERMISSIVE_ACCEPT = 75
POS_ACCEPT = 70
MAX_TOTAL_CANDIDATES = 5000
COMMON_TOKEN_SKIP = 100000


def normalize_name(s: str) -> str:
    if not isinstance(s, str):
        return ''
    s = s.lower()
    s = unicodedata.normalize('NFKD', s)
    s = ''.join([c for c in s if not unicodedata.combining(c)])
    s = s.replace("'", "").replace('"', '').replace('.', '')
    s = ' '.join(s.split())
    return s


def pos_group_from_sb(pos: str) -> str:
    if not isinstance(pos, str):
        return 'UNK'
    p = pos.lower()
    if 'goal' in p or 'keeper' in p or p.startswith('g'):
        return 'GK'
    if any(k in p for k in ['back', 'defend', 'centre back', 'left back', 'right back', 'cb', 'rb', 'lb', 'lwb', 'rwb']):
        return 'DEF'
    if any(k in p for k in ['mid', 'centre', 'cm', 'dm', 'am', 'lm', 'rm']):
        return 'MID'
    if any(k in p for k in ['forward', 'att', 'st', 'cf', 'lw', 'rw', 'wing', 'fw']):
        return 'FWD'
    return 'UNK'


def pos_group_from_fifa(fifa_pos_str: str) -> str:
    if not isinstance(fifa_pos_str, str) or fifa_pos_str.strip() == '':
        return 'UNK'
    toks = [t.strip().lower() for t in fifa_pos_str.split(',')]
    for t in toks:
        if t in ['gk', 'goalkeeper']:
            return 'GK'
    for t in toks:
        if any(k in t for k in ['cb', 'rb', 'lb', 'lwb', 'rwb', 'back', 'def']):
            return 'DEF'
    for t in toks:
        if any(k in t for k in ['cm', 'cdm', 'cam', 'mid', 'central']):
            return 'MID'
    for t in toks:
        if any(k in t for k in ['st', 'cf', 'lw', 'rw', 'lf', 'rf', 'fw', 'att']):
            return 'FWD'
    return 'UNK'


def build_fifa_index():
    if not FIFA_PARQ.exists():
        raise FileNotFoundError('FIFA parquet not found; run ingest_fifa.py first')
    df = pd.read_parquet(FIFA_PARQ)
    name_candidates = (df.get('short_name', '').fillna('') + ' || ' + df.get('long_name', '').fillna('')).astype(str).tolist()
    norms = [normalize_name(x) for x in name_candidates]
    # build lookup frame with stable fifa_id
    if 'sofifa_id' in df.columns:
        lookup = df[['sofifa_id', 'short_name', 'player_positions']].copy().rename(columns={'sofifa_id': 'fifa_id'})
    else:
        lookup = df[['short_name', 'player_positions']].copy()
        lookup['fifa_id'] = lookup.index.astype(str)
    lookup['normalized'] = norms
    token_index = {}
    for idx, norm in enumerate(norms):
        toks = [t for t in norm.split() if len(t) > 1]
        for t in toks:
            token_index.setdefault(t, []).append(idx)
    return lookup, norms, token_index


def run_permissive_pass():
    print('Loading review file...')
    review = pd.read_csv(REVIEW_P, dtype={'player_id_sb': int, 'candidate_fifa_id': object, 'score': float, 'status': object, 'candidate_name': object})
    try:
        accepted = pd.read_csv(ACCEPT_P, dtype={'player_id_sb': int, 'fifa_id': object, 'score': float})
    except Exception:
        accepted = pd.DataFrame(columns=['player_id_sb', 'player_name_sb', 'fifa_id', 'score', 'method'])

    sp = pd.read_parquet(SP_P)
    pid_pos = sp.groupby('player_id_sb')['position'].first().to_dict()

    lookup, fifa_norms, token_index = build_fifa_index()
    print('FIFA candidates:', len(fifa_norms))

    unmatched_mask = review['status']=='unmatched'
    to_process = review.loc[unmatched_mask]
    print('Unmatched rows to process:', len(to_process))

    promoted = []
    updated = 0
    for idx, r in to_process.iterrows():
        sbname = r['player_name_sb']
        n = normalize_name(sbname)
        tokens = [t for t in n.split() if len(t) > 1]
        token_sets = []
        for t in tokens:
            idxs = token_index.get(t, [])
            if len(idxs) > COMMON_TOKEN_SKIP:
                continue
            token_sets.append(set(idxs))
        candidate_idxs = set()
        if token_sets:
            if len(token_sets) > 1:
                candidate_idxs = set.intersection(*token_sets)
            else:
                candidate_idxs = token_sets[0]
        else:
            sorted_tokens = sorted(tokens, key=lambda x: len(token_index.get(x, [])))
            for t in sorted_tokens:
                candidate_idxs.update(token_index.get(t, []))
                if len(candidate_idxs) >= MAX_TOTAL_CANDIDATES:
                    break
        if not candidate_idxs:
            continue
        if len(candidate_idxs) > MAX_TOTAL_CANDIDATES:
            candidate_list = list(candidate_idxs)[:MAX_TOTAL_CANDIDATES]
        else:
            candidate_list = list(candidate_idxs)
        choices = [fifa_norms[i] for i in candidate_list]
        res = process.extractOne(n, choices, scorer=fuzz.token_sort_ratio)
        if not res:
            res = process.extractOne(n, choices, scorer=fuzz.token_set_ratio)
        if not res:
            continue
        best_match, sscore, local_idx = res
        global_idx = candidate_list[local_idx] if isinstance(local_idx, int) else None
        if global_idx is None:
            for i in candidate_list:
                if fifa_norms[i] == best_match:
                    global_idx = i
                    break
        if global_idx is None:
            continue
        lrow = lookup.iloc[global_idx]
        fid = lrow.get('fifa_id')
        fifa_pos = lrow.get('player_positions', '')
        sb_pos = pid_pos.get(int(r['player_id_sb']), None)
        sb_group = pos_group_from_sb(sb_pos) if sb_pos else 'UNK'
        fifa_group = pos_group_from_fifa(fifa_pos)
        # Decide acceptance
        if sscore >= PERMISSIVE_ACCEPT:
            review.at[idx, 'status'] = 'accepted_permissive'
            review.at[idx, 'candidate_fifa_id'] = fid
            review.at[idx, 'candidate_name'] = lrow.get('short_name')
            review.at[idx, 'score'] = int(sscore)
            promoted.append({'player_id_sb': int(r['player_id_sb']), 'player_name_sb': sbname, 'fifa_id': fid, 'score': int(sscore), 'method': 'permissive_fuzzy'})
            updated += 1
        elif sscore >= POS_ACCEPT and sb_group == fifa_group and sb_group != 'UNK':
            review.at[idx, 'status'] = 'accepted_permissive_pos'
            review.at[idx, 'candidate_fifa_id'] = fid
            review.at[idx, 'candidate_name'] = lrow.get('short_name')
            review.at[idx, 'score'] = int(sscore)
            promoted.append({'player_id_sb': int(r['player_id_sb']), 'player_name_sb': sbname, 'fifa_id': fid, 'score': int(sscore), 'method': 'permissive_pos'})
            updated += 1
        elif sscore >= POS_ACCEPT:
            review.at[idx, 'status'] = 'review'
            review.at[idx, 'candidate_fifa_id'] = fid
            review.at[idx, 'candidate_name'] = lrow.get('short_name')
            review.at[idx, 'score'] = int(sscore)
        # progress log
        if updated % 100 == 0 and updated > 0:
            print(f'Promoted so far: {updated}')

    if promoted:
        new_df = pd.DataFrame(promoted)
        accepted = pd.concat([accepted, new_df], ignore_index=True)
        accepted.to_csv(ACCEPT_P, index=False)
        print(f'Appended {len(promoted)} new accepted mappings to {ACCEPT_P}')

    review.to_csv(REVIEW_P, index=False)
    print('Updated review CSV written.')
    print('Done. Promoted:', len(promoted))


if __name__ == '__main__':
    run_permissive_pass()
