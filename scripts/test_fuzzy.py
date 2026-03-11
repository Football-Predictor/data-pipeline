#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
from rapidfuzz import process, fuzz

FIFA_CACHE_P = Path('data') / 'cache' / 'fifa_players.parquet'
df=pd.read_parquet(FIFA_CACHE_P)
names=df['long_name'].astype(str).tolist()
for name in ['Leroy Sané','Sead Kolašinac','Harry Kane','Thomas Müller']:
    res = process.extractOne(name, names, scorer=fuzz.WRatio, score_cutoff=65)
    print(name,'->',res)
