#!/usr/bin/env python3
# /// script
# dependencies = ["pandas","pyarrow"]
# ///
import pandas as pd
p='data/processed/players_train_1000.parquet'
df=pd.read_parquet(p)
print('players table rows=', len(df))
print('columns=', list(df.columns))
print('\nsample:')
print(df.head(8).to_string(index=False))

m='data/processed/matches_train_1000.parquet'
mdf=pd.read_parquet(m)
print('\nmatches table rows=', len(mdf))
print('columns=', list(mdf.columns))
print('\nsample match row (first):')
print(mdf.iloc[0].to_dict())
