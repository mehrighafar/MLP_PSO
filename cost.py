# by Mehri Abdolghafar

from sklearn.neural_network import MLPClassifier as mlpc
from sklearn.metrics import accuracy_score
import config


def cost(*args, **kwargs):
    clf = mlpc(hidden_layer_sizes=(5, 2), activation='tanh',
               solver='sgd', learning_rate_init=args[0][1],
               verbose=True, momentum=args[0][0], early_stopping=False)

    clf.fit(config.train_data_cost, config.train_data_cost_label)
    output = clf.predict(config.test_data_cost)
    err = 1 - accuracy_score(config.test_data_cost_label, output)
    config.model[(args[0][0], args[0][1])] = clf
    return err
