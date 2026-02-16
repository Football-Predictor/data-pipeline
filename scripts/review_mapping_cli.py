#!/usr/bin/env python3
"""Simple CLI to review mapping rows requiring human judgement.

Usage: uv run scripts/review_mapping_cli.py

Commands while viewing a row:
 - a : accept candidate as-is
 - p : accept candidate but mark as position-validated
 - r : mark row as review (leave as-is)
 - u : mark row as unmatched
 - e <num> : set candidate using FIFA candidate index from recent search
 - f <query> : search FIFA names and show top matches
 - q : quit (saves progress)
"""
# /// script
# dependencies = ["pandas","pyarrow","rapidfuzz"]
# ///
from pathlib import Path
import pandas as pd
from rapidfuzz import process, fuzz
import unicodedata

ROOT = Path('data')
REVIEW_P = ROOT / 'mappings' / 'player_map_review.csv'
ACCEPT_P = ROOT / 'mappings' / 'player_map.csv'
FIFA_PARQ = ROOT / 'cache' / 'fifa_players.parquet'


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
    review = pd.read_csv(REVIEW_P)
    fifa = pd.read_parquet(FIFA_PARQ)
    fifa_names = (fifa.get('short_name','').fillna('') + ' || ' + fifa.get('long_name','').fillna('')).astype(str).tolist()
    norms = [normalize_name(x) for x in fifa_names]

    # rows to review: status in review or unmatched
    rows_idx = review[review['status'].isin(['review','unmatched'])].index.tolist()
    if not rows_idx:
        print('No rows to review')
        return
    i = 0
    last_search_choices = []
    while i < len(rows_idx):
        idx = rows_idx[i]
        r = review.loc[idx]
        print('\n---')
        print(f"Row {i+1}/{len(rows_idx)} idx={idx}")
        print('player_id_sb:', r['player_id_sb'])
        print('player_name_sb:', r['player_name_sb'])
        print('current_candidate:', r.get('candidate_name'), r.get('candidate_fifa_id'), 'score=', r.get('score'))
        print('status:', r['status'], 'method:', r.get('method',''))
        cmd = input('[a/p/r/u/f <query>/e <num>/q]/Enter: ').strip()
        if cmd == 'a':
            review.at[idx,'status']='accepted_manual'
            # write to accepted file
            try:
                accepted = pd.read_csv(ACCEPT_P)
            except Exception:
                accepted = pd.DataFrame(columns=['player_id_sb','player_name_sb','fifa_id','score','method'])
            accepted = pd.concat([accepted,pd.DataFrame([{'player_id_sb':int(r['player_id_sb']),'player_name_sb':r['player_name_sb'],'fifa_id':r.get('candidate_fifa_id'),'score':int(r.get('score') or 0),'method':'manual'}])],ignore_index=True)
            accepted.to_csv(ACCEPT_P,index=False)
            i+=1
        elif cmd == 'p':
            review.at[idx,'status']='accepted_manual_pos'
            try:
                accepted = pd.read_csv(ACCEPT_P)
            except Exception:
                accepted = pd.DataFrame(columns=['player_id_sb','player_name_sb','fifa_id','score','method'])
            accepted = pd.concat([accepted,pd.DataFrame([{'player_id_sb':int(r['player_id_sb']),'player_name_sb':r['player_name_sb'],'fifa_id':r.get('candidate_fifa_id'),'score':int(r.get('score') or 0),'method':'manual_pos'}])],ignore_index=True)
            accepted.to_csv(ACCEPT_P,index=False)
            i+=1
        elif cmd == 'r' or cmd=='':
            print('left as review')
            i+=1
        elif cmd == 'u':
            review.at[idx,'status']='unmatched'
            i+=1
        elif cmd.startswith('f '):
            q = cmd[2:].strip()
            n = normalize_name(q)
            res = process.extract(n, norms, scorer=fuzz.token_set_ratio, limit=10)
            last_search_choices = res
            for j,(cand,score,loc) in enumerate(res):
                print(f'[{j}] {fifa_names[loc]} (score={score})')
        elif cmd.startswith('e '):
            try:
                num = int(cmd[2:].strip())
                if 0 <= num < len(last_search_choices):
                    loc = last_search_choices[num][2]
                    cand_name = fifa_names[loc]
                    # extract fifa id
                    fid = fifa.iloc[loc].get('sofifa_id') if 'sofifa_id' in fifa.columns else loc
                    review.at[idx,'candidate_fifa_id']=fid
                    review.at[idx,'candidate_name']=cand_name
                    review.at[idx,'score']=int(last_search_choices[num][1])
                    print('Set candidate to', cand_name, fid)
                else:
                    print('Invalid index or no search yet')
            except Exception as e:
                print('Error parsing selection', e)
        elif cmd == 'q':
            break
        else:
            print('Unknown command')
    # save review df
    review.to_csv(REVIEW_P,index=False)
    print('Saved review file')

if __name__=='__main__':
    main()
