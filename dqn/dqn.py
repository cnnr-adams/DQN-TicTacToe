# Main class, Handles All DQN related things
import random
import numpy as np
from dqn.experience import ExperienceStore
from dqn.q_prediction import QPredictor


class DQN:
    def __init__(self, model_constructor, length_outputs, epsilon=1, epsilon_dec=1.015, epsilon_min=0, gamma=0.25, update_target_iterations=10, save_training=True):
        self.q_predictor = QPredictor(model_constructor())
        target_p = model_constructor()
        target_p.set_weights(self.q_predictor.model.get_weights())
        self.target_predictor = QPredictor(target_p)
        self.model_constructor = model_constructor
        self.experience_store = ExperienceStore(save_training)
        self.last_state_action = None
        self.epsilon = epsilon
        self.update_it = 0
        self.target_iterations = update_target_iterations
        self.epsilon_dec = epsilon_dec
        self.epsilon_min = epsilon_min
        self.gamma = gamma
        self.length_outputs = length_outputs

    def force_epsilon(self, epsilon):
        self.epsilon_min = epsilon
        self.epsilon = epsilon
        self.epsilon_dec = 1

    # Also handles preparing data to be stored
    def determine_action(self, state, reward, terminal_state=False):
        if self.last_state_action:
            self.store_experience(self.last_state_action[0], self.last_state_action[1], reward, state)
        # No more actions needed, don't need to predict
        if terminal_state:
            return -1

        # return random action
        if random.random() < self.epsilon:
            action = random.randint(0, self.length_outputs - 1)
            self.last_state_action = (state, action)
            return action

        # predict q value and return max q action
        action = np.argmax(self.q_predictor.predict(state))
        self.last_state_action = (state, action)
        return action

    def store_experience(self, state, action, reward, next_state):
        self.experience_store.store(state, action, reward, next_state)

    def train(self, epsilon=None):
        self.update_it += 1
        if self.update_it >= self.target_iterations:
            self.target_predictor.model.set_weights(self.q_predictor.model.get_weights())
            self.update_it = 0
        if len(self.experience_store.experience_store) == 0:
            print("experience empty")
            return
        self.epsilon = max(self.epsilon / self.epsilon_dec, self.epsilon_min)
        print("EPSILON:", self.epsilon)
        # Gets 100% of the training data in current store - also removes all the data from current store
        training_data = self.experience_store.get_batch(200)
        # just the states
        training_X = [data_point['state'] for data_point in training_data]

        training_Y = [self.q_predictor.predict(data_point['state']) for data_point in training_data]
        # Changes the Q action value used to closer to 'real' action value
        for i in range(len(training_Y)):
            data_point = training_data[i]
            training_Y[i][data_point['action']] = data_point['reward'] + self.gamma * max(self.target_predictor.predict(data_point['next_state']))
        # for el in zip(training_X, training_Y):
        #    print(el)
        #print(training_X[:10], training_Y[:10])
        self.q_predictor.train(training_X, training_Y)

    def win_chance(self, state):
        return (self.q_predictor.predict(state) + 1) / 2 * 100
