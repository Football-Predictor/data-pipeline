#!/usr/bin/env python3
"""Rollback the last erroneous scrape run.

This will:
  * reset any rows in player_map_review.csv with status 'accepted_scrape'
    back to 'unmatched' and clear candidate fields.
  * drop entries from player_map.csv where method=='scrape' and fifa_id==73580
    (the bogus Kinsk\u00fd candidate) to undo the accidental mappings.
"""
import pandas as pd

REV='data/mappings/player_map_review.csv'
MAP='data/mappings/player_map.csv'

print('loading review...')
r = pd.read_csv(REV, dtype=str)
before = r['status'].value_counts().to_dict()
mask = r['status']=='accepted_scrape'
r.loc[mask, ['candidate_fifa_id','candidate_name','score','method']] = pd.NA
r.loc[mask, 'status'] = 'unmatched'
print('status counts before', before)
print('status counts after', r['status'].value_counts().to_dict())
r.to_csv(REV, index=False)

print('filtering mapping file...')
m = pd.read_csv(MAP, dtype=str)
orig = len(m)
# remove the rows with the bad fifa_id appearing in our erroneous run
m = m[~((m['method']=='scrape') & (m['fifa_id']=='73580'))]
after = len(m)
print('removed', orig-after, 'rows from map.csv')
m.to_csv(MAP, index=False)

print('rollback complete')
