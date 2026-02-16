#!/usr/bin/env python3
# /// script
# dependencies = ["pandas"]
# ///
import pandas as pd
pr = pd.read_csv('data/mappings/player_map_review.csv')
print('Columns:', pr.columns.tolist())
print('Sample:', pr.head(5).to_dict(orient='records'))
