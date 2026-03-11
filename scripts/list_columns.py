#!/usr/bin/env python3
import pandas as pd
p='data/cache/fifa_players.parquet'
df=pd.read_parquet(p)
print(list(df.columns))
