import numpy as np
from collections import deque


class TransitionMemory:
    def __init__(self, state_dim, act_dim, maxlen=20000):
        # shape arguments must be tuple!
        self.data = deque(maxlen=maxlen)
        self.state_dim = state_dim
        self.act_dim = act_dim

    def append(self, state, act, next_state):
        self.data.append((state, act, next_state))

    def sample_batch(self, size):
        # uniform sampling
        # prepare batch containers
        state_batch = np.zeros((size, self.state_dim))
        act_batch = np.zeros((size, self.act_dim))
        next_state_batch = np.zeros((size, self.state_dim))

        num_data = len(self.data)
        rng = np.random.default_rng()
        idxs = rng.choice(num_data, size)

        for i, idx in enumerate(idxs):
            # stack samples
            (state, act, next_state) = self.data[idx]
            state_batch[i], act_batch[i], next_state_batch[i] = state, act, next_state

        return (state_batch, act_batch, next_state_batch)

    def __len__(self):
        return len(self.data)
