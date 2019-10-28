import numpy as np


class QPredictor():
    def __init__(self, model):
        self.model = model

    def predict(self, state):
        pred = self.model.predict(np.array([state]))[0]
        return pred

    def train(self, train_X, train_Y):
        self.model.fit(np.array(train_X), np.array(train_Y), epochs=1, verbose=0)
