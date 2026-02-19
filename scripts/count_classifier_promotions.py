#!/usr/bin/env python3
# /// script
# dependencies = ["pandas"]
# ///
import pandas as pd
from pathlib import Path
pm = Path('data/mappings/player_map.csv')
if not pm.exists():
    print('player_map.csv not found')
    raise SystemExit(1)
df = pd.read_csv(pm)
# count exact method matches and 'contains' as fallback
classifier_exact = df['method'].fillna('').str.strip().value_counts().to_dict()
count_classifier = df['method'].fillna('').str.contains('classifier', na=False).sum()
print('Total rows in player_map.csv:', len(df))
print('Rows where method contains "classifier":', int(count_classifier))
print('\nBreakdown (method -> count):')
for k,v in sorted(classifier_exact.items()):
    if 'classifier' in k:
        print(f' - {k}: {v}')

# Also report how many review rows were promoted in review file
pr = Path('data/mappings/player_map_review.csv')
if pr.exists():
    r = pd.read_csv(pr)
    promoted_review = r[r['status'].str.contains('accepted_classifier', na=False)] if 'status' in r.columns else pd.DataFrame()
    print('\nReview-file rows with status containing "accepted_classifier":', len(promoted_review))
else:
    print('\nplayer_map_review.csv not found')
