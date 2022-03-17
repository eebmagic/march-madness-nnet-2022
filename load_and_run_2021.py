import torch
import torch.nn as nn
import json
import numpy as np
from joblib import load


# # Pytorch
# from train_pytorch import Network
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(f'Using device: {device}')

# model = Network().to(device)
# print(model.parameters)

# model.load_state_dict(torch.load('trained_models/8_epochs_MSE_1e-8.pt', map_location=device))
# model.load_state_dict(torch.load('trained_models/8000000_epochs_MSE_1e-8.pt', map_location=device))
# model.load_state_dict(torch.load('trained_models/1600000_epochs_MSE_1e-06.pt', map_location=device))

# # print(dir(model.hidden))
# # print(model.hidden.weight.size())
# # print(model.hidden.bias.size())

# with open('training_data.json') as file:
#     X, y = json.load(file)
#     y = [1.0 if val else 0.0 for val in y]
#     permutation = np.random.permutation(len(X))
#     X = torch.tensor(X).to(device)
#     X = nn.functional.normalize(X)
#     y = torch.tensor(y).to(device)
#     y = torch.reshape(y, (y.size()[0], 1))


# y_pred = model(X).round()
# print(y_pred)
# print(y_pred.mean())
# print(y_pred.std())

# match = (y_pred == y)
# print(match)
# correct = match.sum().item()
# print(f'{correct=}, {correct}/{len(match)}={correct / len(match)}')

# Sklearn
clf = load('trained_models/sklearn_10_1e-06.joblib')

with open('training_data.json') as file:
    X, y = json.load(file)
    y = np.array([1.0 if val else 0.0 for val in y])

print(clf)

y_pred = np.array([1.0 if val else 0.0 for val in clf.predict(X)])
print(y)
print(y_pred)
match = (y == y_pred)
print(match)
print(match.sum(), match.sum() / len(match))
