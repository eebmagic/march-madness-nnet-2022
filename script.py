import pandas as pd
import numpy as np

regCompact = pd.read_csv('data/mens-march-mania-2022/MDataFiles_Stage1/MRegularSeasonCompactResults.csv')
teamNames = pd.read_csv('data/mens-march-mania-2022/MDataFiles_Stage1/MTeams.csv')

years = regCompact[regCompact.Season >= 2003].Season.unique()


for year in years:
    yearData = regCompact[regCompact.Season == year]
    # print(yearData)
    teamWins = yearData.WTeamID.value_counts()
    teamLosses = yearData.LTeamID.value_counts()
    teams = set(yearData.WTeamID.unique())
    teams = teams.union(set(yearData.LTeamID.unique()))

    totals = {}
    values = []
    zeros = []
    for team in teams:
        value = 0
        if team in teamWins:
            value += teamWins[team]
        if team in teamLosses:
            value += teamLosses[team]
        totals[team] = value
        values.append(value)
        if value == 0:
            zeros.append(team)

    # print(totals)
    # print(f'For year: {year} there were {len(zeros)} teams with no games')
    # print(zeros)

    print(year, min(values), max(values))

