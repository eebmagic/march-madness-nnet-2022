import torch
from torch import nn
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import numpy as np
import json

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

def testAcc(a, b):
    similar = (a == b)
    # print(a[:10])
    # print(b[:10])
    # print(similar[:10])
    # print(similar.sum().item())
    # print(similar.size())
    return similar.sum().item()

with open('training_data.json') as file:
    X, y = json.load(file)
    y = [1.0 if val else 0.0 for val in y]
    permutation = np.random.permutation(len(X))
    X = torch.tensor(X).to(device)
    X = nn.functional.normalize(X)
    y = torch.tensor(y).to(device)
    y = torch.reshape(y, (y.size()[0], 1))
    X = X[permutation]
    y = y[permutation]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    featureSize = X.size()[-1]
    outFeatureSize = y.size()[-1]
    hiddenLayerCount = featureSize * 2
    print('hiddenlayer count: ', hiddenLayerCount)

    # print(X)
    # print(y)
    # print(featureSize)
    # print(outFeatureSize)


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Inputs to hidden layer linear transformation
        print(f'Making Network with vars:')
        print(f'\t{featureSize=}')
        print(f'\t{hiddenLayerCount=}')
        print(f'\t{outFeatureSize=}')
        self.hidden = nn.Linear(featureSize, hiddenLayerCount)
        # Output layer, 10 units - one for each digit
        self.output = nn.Linear(hiddenLayerCount, outFeatureSize)
        
        # Define sigmoid activation and softmax output 
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        # Pass the input tensor through each of our operations
        x = self.hidden(x)
        x = self.sigmoid(x)
        x = self.output(x)
        # x = self.softmax(x)
        
        return x


if __name__ == '__main__':
    model = Network().to(device)

    loss_fn = torch.nn.MSELoss(reduction='sum')
    learning_rate = 1e-6
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    losses = []
    epochs = 1_600_000
    for t in tqdm(range(epochs)):
        # print(f'Running iter {t} ...')

        optimizer.zero_grad()
        y_pred = model(X_train)
        # print(y_pred)
        loss = loss_fn(y_pred, y_train)
        # print(t, loss.item())
        if t % 100 == 0:
            print(f'{t/epochs:.4f}', t, loss.item())
            losses.append(loss.item())
        loss.backward()
        optimizer.step()

    y_pred = model(X_test)
    y_pred = torch.round(y_pred).type(torch.int64)
    y_test = y_test.type(torch.int64)
    # print(y_pred[:10])
    # print(y_test[:10])
    acc = testAcc(y_pred, y_test)
    print(f'{acc} of {y_pred.size()[0]} = {acc / y_pred.size()[0]}')

    torch.save(model.state_dict(), f'trained_models/{epochs}_epochs_MSE_{str(learning_rate)}.pt')

    # print(model.parameters)
    with open('losses.json', 'w') as file:
        json.dump(losses, file)
        print(f'WROTE TO FILE')
