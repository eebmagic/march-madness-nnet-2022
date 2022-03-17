from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from joblib import dump
import numpy as np
import json

def correctCount(A, B):
    Apos = [val > 0 for val in A]
    Bpos = [val > 0 for val in B]

    count = 0
    for a, b in zip(Apos, Bpos):
        if a == b:
            count += 1
    return count

with open('training_data.json') as file:
    X, y = json.load(file)
    permutation = np.random.permutation(len(X))
    X = np.array(X)
    y = np.array(y)
    X = X[permutation]
    y = y[permutation]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


# clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15,))
iters = 3_000
alpha = 1e-6
size = (len(X), len(X) * 2)
clf = MLPClassifier(solver='adam', max_iter=iters, alpha=alpha,
    hidden_layer_sizes=size, random_state=1, verbose=True)
    # hidden_layer_sizes=(len(X), 80), random_state=1, verbose=True)

# print(f'Fitting for {i}th time ...')
print(f'Fitting...')
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
# mse = mean_squared_error(y_test, y_pred)
count = correctCount(y_test, y_pred)
# print(count, count / len(y_test))
# print(y_test)
# print(y_pred)
# print(f'Mean-Squared-Error: {mse}')
print(f'Correct Winner Count: {count}')
print(f'Correct percent: {count / len(y_test)}')

filename = f'trained_models/sklearn_{str(iters)}_{str(alpha)}.joblib'
dump(clf, filename)
print(f'DUMPED TO FILE: {filename}')
