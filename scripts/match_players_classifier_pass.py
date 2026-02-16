#!/usr/bin/env python3
"""Use trained classifier to promote good candidate mappings from 'review' status.
"""
# /// script
# dependencies = ["pandas","joblib","rapidfuzz","scikit-learn"]
# ///
from pathlib import Path
import pandas as pd
import joblib
from rapidfuzz import fuzz
import unicodedata

ROOT = Path('data')
REVIEW_P = ROOT / 'mappings' / 'player_map_review.csv'
ACCEPT_P = ROOT / 'mappings' / 'player_map.csv'
MODEL_P = ROOT / 'models' / 'mapping_classifier.joblib'

THRESH_PROMOTE = 0.90
THRESH_PROMOTE_POS = 0.75


def normalize_name(s: str) -> str:
    if not isinstance(s, str):
        return ''
    s = s.lower()
    s = unicodedata.normalize('NFKD', s)
    s = ''.join([c for c in s if not unicodedata.combining(c)])
    s = s.replace("'", "").replace('"','').replace('.','')
    s = ' '.join(s.split())
    return s


def features(sbn, cn, sb_pos, fifa_pos=''):
    sbn = normalize_name(sbn)
    cn = normalize_name(cn)
    return [
        fuzz.token_sort_ratio(sbn,cn),
        fuzz.token_set_ratio(sbn,cn),
        fuzz.partial_ratio(sbn,cn),
        fuzz.ratio(sbn,cn),
        abs(len(sbn)-len(cn)),
        int(sbn and cn and sbn[0]==cn[0]),
        0 # pos_match unknown unless fifa_pos available
    ]


def run_pass():
    review = pd.read_csv(REVIEW_P)
    try:
        accepted = pd.read_csv(ACCEPT_P)
    except Exception:
        accepted = pd.DataFrame(columns=['player_id_sb','player_name_sb','fifa_id','score','method'])
    if not MODEL_P.exists():
        print('Model not found; train first by running scripts/train_mapping_classifier.py')
        return
    clf = joblib.load(MODEL_P)
    candidates = review[(review['status']=='review') & (pd.notna(review['candidate_fifa_id']))]
    print('Review rows with candidate:', len(candidates))
    promoted=[]
    for idx, r in candidates.iterrows():
        sbn=r['player_name_sb']
        cn=r.get('candidate_name') or ''
        feats=features(sbn,cn,r.get('position',''))
        prob = clf.predict_proba([feats])[0][1]
        if prob >= THRESH_PROMOTE:
            review.at[idx,'status']='accepted_classifier'
            review.at[idx,'score']=int(r.get('score') or 0)
            promoted.append({'player_id_sb':int(r['player_id_sb']),'player_name_sb':r['player_name_sb'],'fifa_id':int(r['candidate_fifa_id']),'score':int(r.get('score') or 0),'method':'classifier'})
        elif prob >= THRESH_PROMOTE_POS:
            review.at[idx,'status']='accepted_classifier_pos'
            review.at[idx,'score']=int(r.get('score') or 0)
            promoted.append({'player_id_sb':int(r['player_id_sb']),'player_name_sb':r['player_name_sb'],'fifa_id':int(r['candidate_fifa_id']),'score':int(r.get('score') or 0),'method':'classifier_pos'})
    if promoted:
        new_df = pd.DataFrame(promoted)
        accepted = pd.concat([accepted,new_df],ignore_index=True)
        accepted.to_csv(ACCEPT_P,index=False)
        review.to_csv(REVIEW_P,index=False)
        print('Appended',len(promoted),'promoted rows')
    else:
        print('No promotions')

if __name__=='__main__':
    run_pass()
