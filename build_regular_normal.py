import pandas as pd
import numpy as np
import json
from tqdm import tqdm

from data_loader import getRegularSeason

np.seterr(divide='ignore', invalid='ignore')

with open('tourney_participant_team_ids.json') as file:
    tourneyYearIds = json.load(file)
    years = list(tourneyYearIds.keys())
    allTeams = set()
    for year in tourneyYearIds:
        allTeams = allTeams.union(set(tourneyYearIds[year]))


dataStarted = False
allData = None
print(f'Building min/max arrays...')
for year in tqdm(tourneyYearIds):
    for team in tourneyYearIds[year]:
        data = getRegularSeason(team, year)

        if True in np.isnan(data):
            print(data)
            print(year, team)
            quit()

        if not dataStarted:
            allData = data
            dataStarted = True
        else:
            try:
                allData = np.vstack((allData, data))
            except ValueError:
                print(f'RAN INTO ERROR')
                print(f'{year=}')
                print(f'{team=}')
                print(f'All data shape: {allData.shape}')
                print(f'Data shape: {data.shape}')
                quit()

mins = allData.min(axis=0)
maxs = allData.max(axis=0)
diffs = maxs - mins

allNormaldata = {}
print(f'Building arrays for file...')
for year in tqdm(tourneyYearIds):
    yearData = {}
    for team in tourneyYearIds[year]:
        data = getRegularSeason(team, year)

        data -= mins
        data = np.divide(data, diffs)
        data = np.nan_to_num(data)

        yearData[str(team)] = data.tolist()
    allNormaldata[str(year)] = yearData

with open('normal_data.json', 'w') as file:
    json.dump(allNormaldata, file)
    print(f'WROTE TO FILE')
