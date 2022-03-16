from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import json

with open('training_data.json') as file:
    X, y = json.load(file)
    # X = np.array(X)
    # y = np.array(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15,))
clf = MLPClassifier(solver='adam', max_iter=20, alpha=1e-10,
    hidden_layer_sizes=(len(X), len(X)*2), random_state=1, verbose=True)
# print(clf)
# quit()

epochs = 1
for i in range(epochs):
    # print(f'Fitting for {i}th time ...')
    print(f'Fitting...')
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(y_test)
    print(y_pred)
    print(mse)
