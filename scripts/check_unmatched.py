#!/usr/bin/env python3
import pandas as pd
r=pd.read_csv('data/mappings/player_map_review.csv',dtype=str)
print(r['status'].value_counts())
print(r[r['status']=='unmatched'].head(10))
