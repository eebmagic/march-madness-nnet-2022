import pandas as pd
import json

ordData = pd.read_csv('data/mens-march-mania-2022/MDataFiles_Stage1/MMasseyOrdinals.csv')

years = ordData[ordData.Season >= 2003].Season.unique()

with open('valid_ordinals.json') as file:
    ords = json.load(file)

for year in years:
    ordCounts = {}
    data = ordData[ordData.Season == year]
    for ordinal in ords:
        subdata = data[data.SystemName == ordinal]
        ordCounts[ordinal] = len(subdata.TeamID.unique())
    print(year, set(ordCounts.values()))
