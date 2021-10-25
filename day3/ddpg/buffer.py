import numpy as np
from collections import namedtuple

class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, limit):
        self.states = Memory(shape=(obs_dim,), limit=limit)
        self.actions = Memory(shape=(act_dim,), limit=limit)
        self.rewards = Memory(shape=(1,), limit=limit)
        self.next_states = Memory(shape=(obs_dim,), limit=limit)
        self.terminals = Memory(shape=(1,), limit=limit)

        self.limit = limit
        self.size = 0

    def append(self, state, action, next_state, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.terminals.append(done)

        self.size = self.states.size

    def sample_batch(self, batch_size):
        # idxs must be synchronized across different types of memories
        rng = np.random.default_rng()
        idxs = rng.choice(self.size, batch_size)

        # get batch from each buffer
        states = self.states.get_batch(idxs)
        actions = self.actions.get_batch(idxs)
        rewards = self.rewards.get_batch(idxs)
        next_states = self.next_states.get_batch(idxs)
        terminal_flags = self.terminals.get_batch(idxs)

        batch = Batch(obs=states, act=actions, next_obs=next_states, rew=rewards, done=terminal_flags)

        return batch


# batch definition as python named tuple
Batch = namedtuple('Batch', ['obs', 'act', 'next_obs', 'rew', 'done'])
    

class Memory:
    """
    implementation of a circular buffer
    """

    def __init__(self, shape, limit=1000000, dtype=np.float):
        self.start = 0
        self.data_shape = shape
        self.size = 0
        self.dtype = dtype
        self.limit = limit
        self.data = np.zeros((self.limit,) + shape)

    def append(self, data):
        if self.size < self.limit:
            self.size += 1
        else:
            self.start = (self.start + 1) % self.limit

        self.data[(self.start + self.size - 1) % self.limit] = data

    def get_batch(self, idxs):

        return self.data[(self.start + idxs) % self.limit]
    
    @property
    def __len__(self):
        return self.size