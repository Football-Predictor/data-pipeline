#!/usr/bin/env python3
# /// script
# dependencies = ["pandas","pyarrow"]
# ///
import pandas as pd
f='data/cache/fifa_players_1000.parquet'
df=pd.read_parquet(f)
print('rows kept in fifa_players_1000.parquet =', len(df))
print('sample short_name head:', df['short_name'].head().tolist())
