import pandas as pd
import json

ordData = pd.read_csv('data/mens-march-mania-2022/MDataFiles_Stage1/MMasseyOrdinals.csv')

systems = ordData.SystemName.unique()
print(ordData)
print(systems)

years = ordData[ordData.Season >= 2003].Season.unique()

yearSystems = {}
smallest = None
smallestCount = float('inf')
for year in years:
    data = ordData[ordData.Season == year]

    systs = set(data.SystemName.unique())
    # print(systs)
    print(year, len(systs))
    yearSystems[year] = systs

    if len(systs) < smallestCount:
        smallest = year
        smallestCount = len(systs)


print(f'\n{smallest, smallestCount}')

systemsSelection = yearSystems[smallest]
for year in years:
    systemsSelection = systemsSelection.intersection(yearSystems[year])

print(f'Final {len(systemsSelection)} systems: {systemsSelection}')

with open('valid_ordinals.json', 'w') as file:
    json.dump(list(systemsSelection), file)
    print(f'WROTE TO FILE')
