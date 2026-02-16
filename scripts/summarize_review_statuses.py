#!/usr/bin/env python3
"""Summarize statuses in player_map_review.csv"""
# /// script
# dependencies = ["pandas"]
# ///
import pandas as pd
pr = pd.read_csv('data/mappings/player_map_review.csv')
counts = pr['status'].value_counts(dropna=False)
print('Status counts:')
print(counts.to_string())
print('\nTotal rows in review file:', len(pr))
print('Rows with status == "review":', (pr['status']=='review').sum())
print('Rows with status starting with "accepted":', pr['status'].astype(str).str.startswith('accepted').sum())
print('Rows with status == "unmatched":', (pr['status']=='unmatched').sum())
# show top 10 'review' rows
print('\nSample review rows (status=="review"):\n')
print(pr.loc[pr['status']=='review'].head(10).to_string(index=False))
