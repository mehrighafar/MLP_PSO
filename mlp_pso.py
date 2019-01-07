#by Mehri Haji Abdolghafar

import numpy as np
import pyswarm
from sklearn.datasets import load_iris
import cost
from sklearn.neural_network import MLPClassifier as mlpc
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import config
import math

data, target = load_iris(True)


train_data, test_data, train_data_label, test_data_label = train_test_split(data, target, test_size=0.3, random_state=0)
config.train_data_cost, config.test_data_cost, config.train_data_cost_label, config.test_data_cost_label = train_test_split(train_data, train_data_label, test_size=0.3, random_state=0)

best_particle, cost_best_particle, BestModel = pyswarm.pso(func = cost.cost, lb = [0.00001,0.00001], ub = [0.99999,0.99999]
                                                , swarmsize=200, debug=True, minstep=math.inf, minfunc=math.inf)

output = BestModel.predict(test_data)

print(f'Accuracy of the Multi layer Perceptron model with the training data is {(1-cost_best_particle)*100:.2f}% '
      f'and Accuracy of this model with the test data is {accuracy_score(test_data_label, output)*100:.2f}%')