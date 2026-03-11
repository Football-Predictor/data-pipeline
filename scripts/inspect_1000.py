#!/usr/bin/env python3
import pandas as pd
p='data/cache/fifa_players_1000.parquet'
df=pd.read_parquet(p)
print(list(df.columns))
print(df.head())
