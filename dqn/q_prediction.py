import numpy as np


class QPredictor():
    def __init__(self, model):
        self.model = model

    def predict(self, state):
        return self.model.predict(np.array([state]))[0]

    def train(self, train_X, train_Y):
        self.model.fit(train_X, train_Y, epochs=1)
