import numpy as np


class QTable:
    def __init__(self, num_states, num_actions, gamma=0.99, pth=None):
        self.gamma = gamma
        if pth is None:
            self.Q = -300. * np.ones(shape=(num_states, num_actions))
        else:
            self.Q = np.load(pth, allow_pickle=True)

    def update(self, state, action, reward, next_state, alpha):
        # update Q-table according to the following update rule:
        # Q(s, a) <- Q(s, a) + alpha * (r + gamma max_{a'} Q(s', a') - Q(s, a))
        target = reward + self.gamma * np.max(self.Q[next_state]) - self.Q[state, action]
        self.Q[state, action] += alpha * target

    def act(self, state):
        return np.argmax(self.Q[state])

    def save(self, pth):
        if pth is None:
            pth = './table.npy'
        np.save(pth, self.Q)

    @property
    def value_ftn(self):
        return np.max(self.Q, axis=1)


