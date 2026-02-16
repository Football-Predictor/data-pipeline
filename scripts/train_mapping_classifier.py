#!/usr/bin/env python3
"""Train a simple classifier to decide whether a candidate mapping is correct.
"""
# /// script
# dependencies = ["pandas","scikit-learn","joblib","rapidfuzz"]
# ///
from pathlib import Path
import pandas as pd
import joblib
from rapidfuzz import fuzz
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import unicodedata

ROOT = Path('data')
REVIEW_P = ROOT / 'mappings' / 'player_map_review.csv'
ACCEPT_P = ROOT / 'mappings' / 'player_map.csv'
MODEL_P = ROOT / 'models' / 'mapping_classifier.joblib'


def normalize_name(s: str) -> str:
    if not isinstance(s, str):
        return ''
    s = s.lower()
    s = unicodedata.normalize('NFKD', s)
    s = ''.join([c for c in s if not unicodedata.combining(c)])
    s = s.replace("'", "").replace('"','').replace('.','')
    s = ' '.join(s.split())
    return s


def feature_row(sb_name, cand_name, sb_pos, fifa_pos):
    sbn = normalize_name(sb_name)
    cn = normalize_name(str(cand_name))
    f = {}
    f['tok_sort']=fuzz.token_sort_ratio(sbn,cn)
    f['tok_set']=fuzz.token_set_ratio(sbn,cn)
    f['partial']=fuzz.partial_ratio(sbn,cn)
    f['ratio']=fuzz.ratio(sbn,cn)
    f['len_diff']=abs(len(sbn)-len(cn))
    f['same_initials']=int(sbn and cn and sbn[0]==cn[0])
    f['pos_match']=0
    try:
        if isinstance(sb_pos,str) and isinstance(fifa_pos,str):
            sbg='UNK'; fbg='UNK'
            if 'goal' in sb_pos.lower(): sbg='GK'
            elif 'def' in sb_pos.lower(): sbg='DEF'
            elif 'mid' in sb_pos.lower(): sbg='MID'
            elif 'att' in sb_pos.lower() or 'st' in sb_pos.lower(): sbg='FWD'
            if 'gk' in fifa_pos.lower(): fbg='GK'
            elif any(k in fifa_pos.lower() for k in ['cb','rb','lb','def']): fbg='DEF'
            elif any(k in fifa_pos.lower() for k in ['cm','mid']): fbg='MID'
            elif any(k in fifa_pos.lower() for k in ['st','cf','fw','att','lw','rw']): fbg='FWD'
            f['pos_match']=int(sbg==fbg and sbg!='UNK')
    except Exception:
        f['pos_match']=0
    return f


def main():
    review = pd.read_csv(REVIEW_P)
    accepted = pd.read_csv(ACCEPT_P)
    # label accepted rows as positive
    accepted_ids = set(accepted['player_id_sb'].astype(int).tolist())
    # build a dataset from review rows where candidate present
    rows = []
    for _, r in review.iterrows():
        if pd.isna(r.get('candidate_fifa_id')):
            continue
        sbn = r['player_name_sb']
        cn = r.get('candidate_name') or ''
        sb_pos = r.get('position')
        # find fifa_pos via candidate id lookup in fifa parquet? skip, use blank
        fifa_pos = ''
        feat = feature_row(sbn, cn, sb_pos, fifa_pos)
        label = 1 if int(r['player_id_sb']) in accepted_ids else 0
        feat['label']=label
        rows.append(feat)
    df = pd.DataFrame(rows)
    if df.empty:
        print('No training rows found; abort')
        return
    X = df[['tok_sort','tok_set','partial','ratio','len_diff','same_initials','pos_match']]
    y = df['label']
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train,y_train)
    preds = clf.predict(X_test)
    print(classification_report(y_test,preds))
    MODEL_P.parent.mkdir(parents=True,exist_ok=True)
    joblib.dump(clf, MODEL_P)
    print('Model written to', MODEL_P)

if __name__=='__main__':
    main()
