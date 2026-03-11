#!/usr/bin/env python3
import pandas as pd
m=pd.read_csv('data/mappings/player_map.csv',dtype=str)
print('total rows', len(m))
print(m.tail(20))
