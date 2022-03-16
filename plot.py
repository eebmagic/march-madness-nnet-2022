import matplotlib.pyplot as plt
import json

with open('losses.json') as file:
    losses = json.load(file)

print(losses)
plt.plot(losses)
plt.show()