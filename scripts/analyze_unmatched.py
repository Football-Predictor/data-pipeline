#!/usr/bin/env python3
"""Analyze unmatched players to see if they appear in FIFA norms or share tokens."""
# /// script
# dependencies = ["pandas", "pyarrow"]
# ///
import pandas as pd
from pathlib import Path
import unicodedata

ROOT = Path('data')
FIFA_PARQ = ROOT / 'cache' / 'fifa_players.parquet'
REVIEW_P = ROOT / 'mappings' / 'player_map_review.csv'


def normalize_name(s: str) -> str:
    if not isinstance(s, str):
        return ''
    s = s.lower()
    s = unicodedata.normalize('NFKD', s)
    s = ''.join([c for c in s if not unicodedata.combining(c)])
    s = s.replace("'", "").replace('"', '').replace('.', '')
    s = ' '.join(s.split())
    return s


def main():
    review = pd.read_csv(REVIEW_P, dtype={'player_id_sb': int, 'player_name_sb': str, 'status': str})
    unmatched = review.loc[review['status']=='unmatched'].copy()
    print('Unmatched count:', len(unmatched))

    # load small subset of fifa normalized names for membership tests efficiently
    print('Loading FIFA normalized names sample...')
    fifa = pd.read_parquet(FIFA_PARQ)
    name_candidates = (fifa.get('short_name', '').fillna('') + ' || ' + fifa.get('long_name', '').fillna('')).astype(str).tolist()
    norms = [normalize_name(x) for x in name_candidates]
    norm_set = set(norms)
    # build token index
    token_index = {}
    for i, n in enumerate(norms):
        toks = [t for t in n.split() if len(t) > 1]
        for t in toks:
            token_index.setdefault(t, 0)
            token_index[t] += 1

    exact_matches = 0
    token_matches = 0
    top_tokens = {}
    for _, r in unmatched.iterrows():
        n = normalize_name(r['player_name_sb'])
        if n in norm_set:
            exact_matches += 1
        toks = [t for t in n.split() if len(t) > 1]
        found = False
        for t in toks:
            if t in token_index:
                token_matches += 1
                found = True
                top_tokens[t] = top_tokens.get(t,0) + 1
                break
    print('Exact normalized matches in FIFA:', exact_matches)
    print('Unmatched players with at least one token present in FIFA names:', token_matches)
    # show top tokens
    top = sorted(top_tokens.items(), key=lambda x: -x[1])[:20]
    print('Top tokens found among unmatched (token,count):')
    print(top)

if __name__ == '__main__':
    main()
