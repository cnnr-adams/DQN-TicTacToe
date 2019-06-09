# Main class, Handles All DQN related things
import random
import numpy as np

from dqn.agent import Agent
from dqn.experience import ExperienceStore
from dqn.q_prediction import QPredictor


class DQN:
    def __init__(self, model, length_outputs, epsilon=0.1, gamma=0.9, save_training=True):
        self.q_predictor = QPredictor(model)
        self.experience_store = ExperienceStore(save_training)
        self.last_state_action = None
        self.epsilon = epsilon
        self.gamma = gamma
        self.length_outputs = length_outputs

    # Also handles preparing data to be stored
    def determine_action(self, state, reward, terminal_state=False):
        if self.last_state_action:
            self.store_experience(self.last_state_action[0], self.last_state_action[1], reward, state)
        # No more actions needed, don't need to predict
        if terminal_state:
            return -1

        # return random action
        if random.random() < self.epsilon:
            return random.randint(0, self.length_outputs - 1)

        # predict q value and return max q action
        action = np.argmax(self.q_predictor.predict(state))
        self.last_state_action = (state, action)
        return action

    def store_experience(self, state, action, reward, next_state):
        self.experience_store.store(state, action, reward, next_state)

    def train(self):
        # Gets 100% of the training data in current store - also removes all the data from current store
        training_data = self.experience_store.get_batch(1)

        # just the states
        training_X = [data_point['state'] for data_point in training_data]

        training_Y = [self.q_predictor.predict(data_point['state']) for data_point in training_data]

        # Changes the Q action value used to closer to 'real' action value
        for i in range(len(training_Y)):
            data_point = training_data[i]
            training_Y[i][data_point['action']] = data_point['reward'] + self.gamma * max(self.q_predictor.predict(data_point['next_state']))

        training_Y = [data_point['reward'] + self.gamma * max(self.q_predictor.predict(data_point['next_state'])) for data_point in training_data]

        self.q_predictor.train(training_X, training_Y)
