import pandas as pd
import json

'''
Get all team ids for teams that play in any march madness tourney 2003-current
'''

tournCompact = pd.read_csv('data/mens-march-mania-2022/MDataFiles_Stage2/MNCAATourneyCompactResults.csv')
teamNames = pd.read_csv('data/mens-march-mania-2022/MDataFiles_Stage2/MTeams.csv')

years = tournCompact[tournCompact.Season >= 2003].Season.unique()
# years = tournCompact[tournCompact.Season >= 2021].Season.unique()

def getName(targetID):
    return teamNames[teamNames.TeamID == targetID].TeamName.tolist()[0]


def getTourneyTeamIDS():
    teamYears = {}
    for year in years:
        data = tournCompact[tournCompact.Season == year]
        teamWins = data.WTeamID.value_counts()

        teamIds = set(data.WTeamID.unique())
        teamIds = teamIds.union(set(data.LTeamID.unique()))

        wins = {team: 0 for team in teamIds}
        for team in teamIds:
            if team in teamWins:
                wins[team] = teamWins[team]

        teamYears[int(year)] = [int(t) for t in list(teamIds)]

        # print(year, len(teamIds))
        # print(wins)

        teams = sorted(list(teamIds), key=lambda t: wins[t])

        # x = getName(teams[3])
        # print(type(x), x)
        # for team in teams:
        #     print(f'{year} | {team} | wins: {wins[team]} | {getName(team)}')

    # print(teamYears)
    return teamYears


if __name__ == '__main__':
    with open('tourney_participant_team_ids.json', 'w') as file:
        json.dump(getTourneyTeamIDS(), file)
        print('WROTE TO FILE')
