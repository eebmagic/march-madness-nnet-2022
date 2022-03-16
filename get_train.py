import pandas as pd
import random
import json
from tqdm import tqdm

from data_loader import getTrainingData
from build_yearly_tourney_ids import getName

tournCompact = pd.read_csv('data/mens-march-mania-2022/MDataFiles_Stage1/MNCAATourneyCompactResults.csv')
tournDetails = pd.read_csv('data/mens-march-mania-2022/MDataFiles_Stage1/MNCAATourneyDetailedResults.csv')

years = [year for year in tournDetails.Season.unique() if year > 2003]

def getTargetPoints():
    '''
    Build sets:
    [
        {
            year: 2004,
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
            # random.shuffle(data)
            point = {'year': int(year), 'day':row.DayNum, 'data': data}
            sets.append(point)

    return sets


def getFullData():
    '''
    For each point in target points:
        get point
        get data for teamA
        get data for teamB

        build input/output
        build inverse

    return
    '''
    X = []
    y = []
    counter = 0
    for point in tqdm(getTargetPoints()):
        teamA = point['data'][0][0]
        teamB = point['data'][1][0]
        teamATarget = point['data'][0][1]
        teamBTarget = point['data'][1][1]
        day = [point['day']]
        year = point['year']

        teamAData = getTrainingData(teamA, year)
        teamBData = getTrainingData(teamB, year)

        xOneIn = day + teamAData + teamBData
        # xOneOut = [teamATarget, teamBTarget]
        xOneOut = teamATarget - teamBTarget

        xTwoIn = day + teamBData + teamAData
        # xTwoOut = [teamBTarget, teamATarget]
        xTwoOut = teamBTarget - teamATarget

        X.append(xOneIn)
        y.append(xOneOut > 0)
        X.append(xTwoIn)
        y.append(xTwoOut > 0)

        # print(xOneIn)
        # print(xOneOut)
        # print(f'For year {year} and teams {getName(teamA)} vs. {getName(teamB)}')

        # if counter > 100:
        #     break
        counter += 1
    
    return X, y

if __name__ == '__main__':
    # points = getTargetPoints()
    data = getFullData()
    with open('training_data.json', 'w') as file:
        json.dump(data, file)
        print('WROTE TO FILE')

