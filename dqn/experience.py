import numpy as np


class ExperienceStore:
    def __init__(self, save_training):
        # TODO: Implement save_training
        self.save_training = save_training
        self.experience_store = []

    def store(self, state, action, reward, next_state):
        self.experience_store.append({
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state
        })

    def get_batch(self, size):
        np.random.shuffle(self.experience_store)
        batch = self.experience_store[:size]
        # reset the experience store for now
        self.experience_store = []
        return batch
