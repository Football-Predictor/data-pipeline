#!/usr/bin/env python3
"""Initial+Surname fuzzy pass: handle cases like 'J. Martinez' or 'J Martinez'.
"""
# /// script
# dependencies = ["pandas","pyarrow","rapidfuzz"]
# ///
from pathlib import Path
import pandas as pd
from rapidfuzz import process, fuzz
import unicodedata

ROOT = Path('data')
FIFA_PARQ = ROOT / 'cache' / 'fifa_players.parquet'
REVIEW_P = ROOT / 'mappings' / 'player_map_review.csv'
ACCEPT_P = ROOT / 'mappings' / 'player_map.csv'
SP_P = ROOT / 'cache' / 'matches_starting_players.parquet'

MAX_CANDIDS = 8000
ACCEPT_SCORE = 75
REVIEW_SCORE = 70


def normalize_name(s: str) -> str:
    if not isinstance(s, str):
        return ''
    s = s.lower()
    s = unicodedata.normalize('NFKD', s)
    s = ''.join([c for c in s if not unicodedata.combining(c)])
    s = s.replace("'", "").replace('"','').replace('.','')
    s = ' '.join(s.split())
    return s


def tokens_of(s: str):
    n = normalize_name(s)
    return [t for t in n.split() if len(t)>0]


def build_index():
    df = pd.read_parquet(FIFA_PARQ)
    name_candidates = (df.get('short_name','').fillna('') + ' || ' + df.get('long_name','').fillna('')).astype(str).tolist()
    norms = [normalize_name(x) for x in name_candidates]
    token_index = {}
    for idx, n in enumerate(norms):
        toks = [t for t in n.split() if len(t)>1]
        for t in toks:
            token_index.setdefault(t, []).append(idx)
    if 'sofifa_id' in df.columns:
        lookup = df[['sofifa_id','short_name','player_positions']].copy().rename(columns={'sofifa_id':'fifa_id'})
    else:
        lookup = df[['short_name','player_positions']].copy(); lookup['fifa_id']=lookup.index.astype(str)
    lookup['normalized']=norms
    return lookup, norms, token_index


def run_initial_surname_pass():
    review = pd.read_csv(REVIEW_P, dtype={'player_id_sb': int})
    # coerce candidate columns to safe dtypes for in-place updates
    if 'candidate_fifa_id' not in review.columns:
        review['candidate_fifa_id'] = pd.Series([pd.NA]*len(review), dtype=object)
    else:
        review['candidate_fifa_id'] = review['candidate_fifa_id'].where(pd.notnull(review['candidate_fifa_id']), pd.NA).astype(object)
    if 'candidate_name' not in review.columns:
        review['candidate_name'] = pd.Series([pd.NA]*len(review), dtype=object)
    else:
        review['candidate_name'] = review['candidate_name'].where(pd.notnull(review['candidate_name']), pd.NA).astype(object)
    if 'score' not in review.columns:
        review['score'] = pd.Series([pd.NA]*len(review), dtype='Int64')
    else:
        review['score'] = review['score'].where(pd.notnull(review['score']), pd.NA).astype('Int64')

    try:
        accepted = pd.read_csv(ACCEPT_P, dtype={'player_id_sb': int})
    except Exception:
        accepted = pd.DataFrame(columns=['player_id_sb','player_name_sb','fifa_id','score','method'])

    sp = pd.read_parquet(SP_P)
    pid_pos = sp.groupby('player_id_sb')['position'].first().to_dict()

    lookup, norms, token_index = build_index()
    unmatched = review.loc[review['status']=='unmatched']
    print('Unmatched rows:', len(unmatched))
    promoted=[]
    for idx, r in unmatched.iterrows():
        toks = tokens_of(r['player_name_sb'])
        if not toks:
            continue
        if len(toks)==1:
            fname=''; sname=toks[0]
            initial=''
        else:
            initial = toks[0][0]
            sname = toks[-1]
        cand_idxs = token_index.get(sname, [])
        if not cand_idxs or len(cand_idxs) > MAX_CANDIDS:
            continue
        # filter candidates whose normalized name has first token starting with initial
        filtered = []
        for i in cand_idxs:
            n = norms[i]
            parts = n.split()
            if not parts:
                continue
            if initial and parts[0].startswith(initial):
                filtered.append(i)
        if not filtered:
            # fallback: use entire cand_idxs but limit size
            filtered = cand_idxs[:MAX_CANDIDS]
        choices = [norms[i] for i in filtered]
        n = normalize_name(r['player_name_sb'])
        res = process.extractOne(n, choices, scorer=fuzz.token_sort_ratio)
        if not res:
            res = process.extractOne(n, choices, scorer=fuzz.token_set_ratio)
        if not res:
            continue
        best_match, sscore, local_idx = res
        global_idx = filtered[local_idx] if isinstance(local_idx,int) else None
        if global_idx is None:
            for i in filtered:
                if norms[i]==best_match:
                    global_idx=i; break
        if global_idx is None:
            continue
        lrow = lookup.iloc[global_idx]
        fid = lrow.get('fifa_id')
        if sscore >= ACCEPT_SCORE:
            review.at[idx,'status']='accepted_initial_surname'
            review.at[idx,'candidate_fifa_id']=fid
            review.at[idx,'candidate_name']=lrow.get('short_name')
            review.at[idx,'score']=int(sscore)
            promoted.append({'player_id_sb':int(r['player_id_sb']),'player_name_sb':r['player_name_sb'],'fifa_id':fid,'score':int(sscore),'method':'initial_surname'})
        elif sscore >= REVIEW_SCORE:
            review.at[idx,'status']='review'
            review.at[idx,'candidate_fifa_id']=fid
            review.at[idx,'candidate_name']=lrow.get('short_name')
            review.at[idx,'score']=int(sscore)

    if promoted:
        new_df = pd.DataFrame(promoted)
        accepted = pd.concat([accepted,new_df],ignore_index=True)
        accepted.to_csv(ACCEPT_P,index=False)
        print(f'Appended {len(promoted)} promoted rows')
    review.to_csv(REVIEW_P,index=False)
    print('Done. promoted:', len(promoted))

if __name__=='__main__':
    run_initial_surname_pass()
