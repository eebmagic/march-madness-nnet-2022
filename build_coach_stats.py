import pandas as pd
import numpy as np
import json

coachData = pd.read_csv('data/mens-march-mania-2022/MDataFiles_Stage2/MTeamCoaches.csv')
regCompact = pd.read_csv('data/mens-march-mania-2022/MDataFiles_Stage2/MRegularSeasonCompactResults.csv')

with open('tourney_participant_team_ids.json') as file:
    tourneyYearIds = json.load(file)
    years = list(tourneyYearIds.keys())
    if str(2022) not in years:
        years.append(str(2022))
    allTeams = set()
    for year in tourneyYearIds:
        allTeams = allTeams.union(set(tourneyYearIds[year]))


def coachCounts():
    '''
    Build coach count per year per team
    '''
    coachCounts = {}
    for team in allTeams:
        yearCounts = {}
        for year in years:
            yearData = coachData.query(f'Season == {year} & TeamID == {team}')
            yearCounts[year] = len(yearData)
        coachCounts[team] = yearCounts

    with open('coach_counts.json', 'w') as file:
        json.dump(coachCounts, file)


def getTeamStanding(teamID, year):
    '''
    Get the total wins and total games played by a team for a given season
    '''
    wins = len(regCompact.query(f'WTeamID == {teamID} & Season == {year}'))
    losses = len(regCompact.query(f'LTeamID == {teamID} & Season == {year}'))

    return wins, wins+losses


def coachStats():
    '''
    Find first year each coach started coaching,
    starting year with each team,
    and win rate.
    
    Output:
    {
        "coach_name": {
            "start": 1990,
            "teamStarts": {
                "1142": 1990,
                "1435": 1996
            },
            "totalWins": 25,
            "totalGames": 45
        }
    }
    '''
    allCoaches = coachData[coachData.TeamID.isin(allTeams)].CoachName.unique()
    stats = {}
    for coach in list(allCoaches)[::-1]:
        data = coachData[coachData.CoachName == coach]
        teamStarts = {}
        for team in data.TeamID.unique():
            subdata = data.query(f'TeamID == {team}')
            teamStarts[str(team)] = min(subdata.Season)
        overallStart = min(teamStarts.values())

        totalWins = 0
        totalGames = 0
        for index, row in data.iterrows():
            wins, games = getTeamStanding(row.TeamID, row.Season)
            totalWins += wins
            totalGames += games

        stats[coach] = {
            "start": overallStart,
            "teamStarts": teamStarts,
            "totalWins": totalWins,
            "totalGames": totalGames
        }

    with open('coach_stats.json', 'w') as file:
        json.dump(stats, file)


def coachTeams():
    allTeamYears = {}
    for team in list(allTeams)[::-1]:
        teamYears = {}
        teamData = coachData[coachData.TeamID == team]
        for year in years:
            yearData = teamData[teamData.Season == int(year)].reset_index()
            if len(yearData) > 0:
                lastCoach = yearData.iloc[yearData.LastDayNum.idxmax()].CoachName
                teamYears[str(year)] = lastCoach
        allTeamYears[str(team)] = teamYears

    with open('coach_from_team_year.json', 'w') as file:
        json.dump(allTeamYears, file)

if __name__ == '__main__':
    coachCounts()
    coachStats()
    coachTeams()
