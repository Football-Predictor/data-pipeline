#!/usr/bin/env python3
import pandas as pd
r=pd.read_csv('data/mappings/player_map_review.csv',dtype=str)
accepted = r[r['status'].str.startswith('accepted')]
print('review status counts')
print(r['status'].value_counts())
m=pd.read_csv('data/mappings/player_map.csv',dtype=str)
print('map rows', len(m))
# appearance coverage
a=pd.read_parquet('data/processed/players_train_1000.parquet')
# not very accurate global but just for 1k
print('players_train_1000 count', len(a))
