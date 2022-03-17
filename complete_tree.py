import torch
import torch.nn as nn
from torch.nn.functional import normalize
import json
import numpy as np
import random
from joblib import load

from train_pytorch import Network
from data_loader import getTrainingData
from data_loader import getSeed
from build_yearly_tourney_ids import getName

# Load pytorch model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Network().to(device)
model.load_state_dict(torch.load('trained_models/8000000_epochs_MSE_1e-8.pt', map_location=device))

# Load sklearn model
clf = load('trained_models/sklearn_10_1e-06.joblib')


def getData(teamAID, teamBID, day):
    '''
    Build array of data for A vs B
    also build array for B vs A as to verify agreement
    '''
    pass

    a = getTrainingData(teamAID, 2022)
    b = getTrainingData(teamBID, 2022)

    # print(a)
    # print(b)

    one = [day] + a + b
    two = [day] + b + a

    return one, two


def sklearnModel(teamAID, teamBID, day):
    one, two = getData(teamAID, teamBID, day)

    oneResult = clf.predict(np.array(one).reshape(1, len(one)))[0]
    twoResult = clf.predict(np.array(two).reshape(1, len(two)))[0]

    print(oneResult)
    print(twoResult)

    aSeed = getSeed(teamAID, 2022)
    bSeed = getSeed(teamBID, 2022)
    if oneResult and not twoResult:
        print(f'{getName(teamAID)} ({aSeed}) BEATS {getName(teamBID)} ({bSeed})')
        return True

    if not oneResult and twoResult:
        print(f'{getName(teamBID)} ({bSeed}) BEATS {getName(teamAID)} ({aSeed})')
        return False

    print('THERE WAS A DISAGREEMENT')
    if aSeed <= bSeed:
        print(f'{getName(teamAID)} ({aSeed}) BEATS {getName(teamBID)} ({bSeed})')
        return True
    else:
        print(f'{getName(teamBID)} ({bSeed}) BEATS {getName(teamAID)} ({aSeed})')
        return False


def pytorchModel(teamAID, teamBID, day, normalizeIns=True):
    '''
    Plug into pytorch model (loaded above).

    Return:
        True: if team A would win
        False if team B would win
    '''
    one, two = getData(teamAID, teamBID, day)
    oneTensor = torch.tensor(one).reshape(1, len(one))
    twoTensor = torch.tensor(two).reshape(1, len(two))
    if normalizeIns:
        oneTensor = normalize(oneTensor)
        twoTensor = normalize(twoTensor)

    print(oneTensor)
    print(twoTensor)

    oneResult = int(model(oneTensor).round().item())
    twoResult = int(model(twoTensor).round().item())
    oneResult = int(model(oneTensor).round().item())
    twoResult = int(model(twoTensor).round().item())
    print(oneResult)
    print(twoResult)

    aSeed = getSeed(teamAID, 2022)
    bSeed = getSeed(teamBID, 2022)
    if oneResult == 1 and twoResult == 0:
        print(f'{getName(teamAID)} ({aSeed}) BEATS {getName(teamBID)} ({bSeed})')
        return True

    if oneResult == 0 and twoResult == 1:
        print(f'{getName(teamBID)} ({bSeed}) BEATS {getName(teamAID)} ({aSeed})')
        return False

    print('THERE WAS A DISAGREEMENT')
    if aSeed <= bSeed:
        print(f'{getName(teamAID)} ({aSeed}) BEATS {getName(teamBID)} ({bSeed})')
        return True
    else:
        print(f'{getName(teamBID)} ({bSeed}) BEATS {getName(teamAID)} ({aSeed})')
        return False


def getPairings():
    with open('data/pairings.txt') as file:
        ptext = file.read().strip()

    lines = [line for line in ptext.split('\n') if line]

    out = []
    for line in lines:
        if line.startswith('#'):
            # print('pass')
            # print('\n', line)
            pass
        else:
            # print(line)
            nums = [int(val) for val in line.split(', ')]
            a = getName(nums[0])
            b = getName(nums[1])
            # print([a, b, nums[2:]])
            # print([nums[0], nums[1], nums[2:]])
            out.append([nums[0], nums[1], nums[2:] + [152, 154]])

    return out


def buildTree(pairings, func):
    '''
    Build a tree with a given decider function
    '''
    total = len(pairings)
    tree = [pairings]
    layer = 0
    while layer <= 5:
        nextLayer = []
        for i, (teamA, teamB, days) in enumerate(tree[-1]):
            wins = func(teamA, teamB, days[layer])
            winner = teamA if wins else teamB
            if i % 2 == 0:
                part = [winner]
            else:
                part.append(winner)
                part.append(days)
                nextLayer.append(part)

            if len(tree[-1]) == 1:
                nextLayer.append(part)

        # print(layer, len(tree[-1]), nextLayer)

        tree.append(nextLayer)
        total = len(nextLayer)
        # break
        layer += 1

    return tree


def treeToString(tree):
    '''
    Convert a tree (array of games and days) to a human readable string.
    '''
    out = ''
    layer = 0
    while layer <= 5:
        out += f'\nLAYER {layer}:\n'
        for i, game in enumerate(tree[layer]):
            a, b, _ = game
            teamA = getName(a)
            teamB = getName(b)

            if i % 2 == 0:
                winner = tree[layer+1][i//2][0]
            else:
                winner = tree[layer+1][i//2][1]
            winner = getName(winner)

            out += f'{teamA} vs {teamB} => {winner}\n'

        layer += 1
    return out


def randSelect(teamAID, teamBID, day):
    '''
    A random function to simulate a model output.
    Used for testing buildTree().
    '''
    return random.choice([True, False])



if __name__ == '__main__':

    # tree = buildTree(getPairings(), randSelect)
    # print(treeToString(tree))

    # pytorchModel(1103, 1417, 136)


    # tree = buildTree(getPairings(), pytorchModel)
    tree = buildTree(getPairings(), sklearnModel)
    print(treeToString(tree))
