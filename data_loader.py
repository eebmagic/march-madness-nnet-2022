import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import numpy as np
import json

from build_yearly_tourney_ids import getName

pd.options.mode.chained_assignment = None

regCompact = pd.read_csv('data/mens-march-mania-2022/MDataFiles_Stage1/MRegularSeasonCompactResults.csv')
regDetailed = pd.read_csv('data/mens-march-mania-2022/MDataFiles_Stage1/MRegularSeasonDetailedResults.csv')
seeds = pd.read_csv('data/mens-march-mania-2022/MDataFiles_Stage1/MNCAATourneySeeds.csv')
ordData = pd.read_csv('data/mens-march-mania-2022/MDataFiles_Stage1/MMasseyOrdinals.csv')

with open('valid_ordinals.json') as file:
    validOrdinals = json.load(file)

with open('coach_counts.json') as file:
    coachCounts = json.load(file)

with open('coach_stats.json') as file:
    coachStats = json.load(file)

with open('coach_from_team_year.json') as file:
    teamCoachByYear = json.load(file)

with open('normal_data.json') as file:
    regularNormalized = json.load(file)

# TODO: Replace FILL values (will require minor stats checks)
FILL = None
RANGES = {
    'seed': [1, 16],
    'ordinalRank': [FILL, FILL],
    'coachCount': [1, 2],
    'coachExperience': [0, FILL],
    'coachTeamExperience': [0, FILL],
    'coachWins': [0, FILL],
    'coachGames': [0, FILL],
    'coachPercent': [0, 1]
}


def getSeed(teamID, year):
    seedData = seeds[seeds.Season == year]
    seedData = seedData[seedData.TeamID == teamID].Seed.tolist()
    if len(seedData) > 0:
        seedData = seedData[0][:3]
        divLetter = seedData[0]
        rank = int(seedData[1:])
    else:
        divLetter = 'None'
        rank = 99

    # seed = ([divLetter=='W', divLetter=='X', divLetter=='Y', divLetter=='Z'], rank)

    return rank


def getRankings(teamID, year):
    out = []
    yearData = ordData[ordData.Season == year]
    teamData = yearData[yearData.TeamID == teamID]
    for ordinal in validOrdinals:
        data = teamData[teamData.SystemName == ordinal]
        data = data[data.RankingDayNum == 133]
        out.append(int(data.OrdinalRank.tolist()[0]))

    return out


def getCoachStats(teamID, year):
    '''
    Get statistics for current coach from json files
    NOTE: These win stats are from total games ever, not before given year...
    '''
    changes = coachCounts[str(teamID)][str(year)]
    name = teamCoachByYear[str(teamID)][str(year)]
    stats = coachStats[name]
    experience = year - stats['start']
    timeWithTeam = year - stats['teamStarts'][str(teamID)]
    totalWins = stats['totalWins']
    totalGames = stats['totalGames']
    winPercent = totalWins / totalGames
    out = [changes, experience, timeWithTeam, totalWins, totalGames, winPercent]

    return out


def getRegularSeason(teamID, year):
    regularGames = regDetailed.query(f'(WTeamID == {teamID} | LTeamID == {teamID}) & Season == {year}')

    # Get seeds for WTeamID and LTeamID before splitting
    WTeamSeeds = [getSeed(team, year) for team in regularGames.WTeamID]
    LTeamSeeds = [getSeed(team, year) for team in regularGames.LTeamID]
    regularGames['WTeamSeed'] = WTeamSeeds
    regularGames['LTeamSeeds'] = LTeamSeeds

    wins = regularGames.query(f'WTeamID == {teamID}')
    losses = regularGames.query(f'LTeamID == {teamID}')

    # Drop cols that shouldn't get stats (Season, TeamID)
    dropableCols = ['Season', 'WTeamID', 'LTeamID', 'DayNum']
    wins = wins.drop(columns=dropableCols)
    losses = losses.drop(columns=dropableCols)

    # Get average, median, and stddev for each col for both sets
    zeros = np.zeros(wins.shape[1]-1)
    winMeans = wins.mean(axis=0)
    winStds = wins.std(axis=0)
    winMeds = wins.median(axis=0)
    if len(wins) == 0:
        winMeans = zeros
        winStds = zeros
        winMeds = zeros
    elif len(wins) == 1:
        winStds = zeros

    lossMeans = losses.mean(axis=0)
    lossStds = losses.std(axis=0)
    lossMeds = losses.median(axis=0)
    if len(losses) == 0:
        lossMeans = zeros
        lossStds = zeros
        lossMeds = zeros
    elif len(losses) == 1:
        lossStds = zeros

    out = winMeans.tolist() + winStds.tolist() + winMeds.tolist()
    out += lossMeds.tolist() + lossStds.tolist() + lossMeds.tolist()

    return np.array(out)


def getNormalizedRegularSeason(teamID, year):
    return regularNormalized[str(year)][str(teamID)]


def getTrainingData(teamID, year, normalized=False):
    '''
    Should include:
        - seed ranking
        - rankings from ordinals
        - coach stats
            - coach changes for team in given year
            - coach career length
            - coach time with team
            - coach wins
            - coach total games
        - regular season performance 
        - past season performance (check if had a past season)
    '''
    assert(year <= 2003, 'Year must be at least AFTER 2003')

    seed = [getSeed(teamID, year)]
    rankings = getRankings(teamID, year)
    currCoachStats = getCoachStats(teamID, year)
    regularStats = getNormalizedRegularSeason(teamID, year)

    if str(teamID) in regularNormalized[str(year-1)]:
        previousStats = getNormalizedRegularSeason(teamID, year-1)
    else:
        previousStats = np.zeros(regularStats.shape)

    # print(seed)
    # print(rankings)
    # print(currCoachStats)
    # print(regularStats)
    # print(previousStats)

    out = seed + rankings + currCoachStats + regularStats + previousStats
    return out


if __name__ == '__main__':
    team = 1386
    year = 2004
    result = getTrainingData(team, year)
