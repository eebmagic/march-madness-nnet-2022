import pandas as pd
import random

tournCompact = pd.read_csv('data/mens-march-mania-2022/MDataFiles_Stage1/MNCAATourneyCompactResults.csv')
tournDetails = pd.read_csv('data/mens-march-mania-2022/MDataFiles_Stage1/MNCAATourneyDetailedResults.csv')

years = [year for year in tournDetails.Season.unique() if year > 2003]

def getTrainPoints():
    '''
    Build sets:
    [
        {
            year: 2004
            day: 120,
            data (shuffled): [
                [1163, 82],
                [1210, 73]
            ]
        }
    ]
    '''
    sets = []
    for year in years:
        yearData = tournCompact.query(f'Season == {year}')

        for _, row in yearData.iterrows():
            data = [[row.WTeamID, row.WScore], [row.LTeamID, row.LScore]]
            random.shuffle(data)
            point = {'year': int(year), 'data': data}
            sets.append(point)

    return sets

