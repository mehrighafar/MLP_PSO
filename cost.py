from sklearn.neural_network import MLPClassifier as mlpc
import numpy as np
from sklearn.metrics import accuracy_score
import config

def cost(x, *args, **kwargs):
    clf = mlpc(hidden_layer_sizes=(5,2), activation='tanh', solver='sgd', learning_rate_init=x[1],
               verbose=True, momentum=x[0], early_stopping=True)

    clf.fit(config.train_data_cost, config.train_data_cost_label)
    output = clf.predict(config.test_data_cost)
    err = 1 - accuracy_score(config.test_data_cost_label, output)
    
    return err, clf