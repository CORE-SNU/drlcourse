import numpy as np
import torch
from torch.optim import Adam
from torch.nn import MSELoss
from model import TransitionModel
from memory import TransitionMemory
import torch.multiprocessing as mp

class ModelBasedAgent:
    def __init__(self,
                 state_dim,
                 act_dim,
                 ctrl_range,
                 hidden1=400,
                 hidden2=400,
                 lr=0.001
                 ):
        self.dimS = state_dim
        self.dimA = act_dim
        self.ctrl_range = ctrl_range
        self.model = TransitionModel(state_dim, act_dim, hidden1, hidden2)
        self.memory = TransitionMemory(state_dim, act_dim)

        self.optimizer = Adam(self.model.parameters(), lr=lr)

    def train(self, batch_size):
        self.model.train()
        # note that training of the dynamics does not depend on any reward info
        (state_batch, act_batch, next_state_batch) = self.memory.sample_batch(batch_size)

        state_batch = torch.tensor(state_batch).float()
        act_batch = torch.tensor(act_batch).float()
        next_state_batch = torch.tensor(next_state_batch).float()

        prediction = self.model(state_batch, act_batch)


        loss_ftn = MSELoss()
        loss = loss_ftn(prediction, next_state_batch)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        loss_val = loss.detach().numpy()

        return loss_val

    def execute_action(self, state, rew_ftn, K, H):
        """
        # generate K trajectories using the model of dynamics and random action sampling, and perform MPC
        # Remark! K roll-outs can be done simultaneously!

        given a state, execute an action based on random-sampling shooting method
        :param state: current state(numpy array)
        :param rew_ftn: vectorized reward function
        :param K: number of candidate action sequences to generate
        :param H: length of time horizon
        :return: action to be executed(numpy array)
        """
        assert K > 0 and H > 0

        dimA = self.dimA

        self.model.eval()

        states = np.tile(state, (K, 1)) # shape = (K, dim S)
        scores = np.zeros(K)    # array which contains cumulative rewards of roll-outs

        # generate K random action sequences of length H
        action_sequences = self.ctrl_range * (2. * np.random.rand(H, K, dimA) - 1.)
        first_actions = action_sequences[0]     # shape = (K, dim A)


        for t in range(H):
            actions = action_sequences[t]    # set of K actions, shape = (K, dim A)
            scores += rew_ftn(states, actions)

            s = torch.tensor(states).float()
            a = torch.tensor(actions).float()

            next_s = self.model(s, a)

            # torch tensor to numpy array
            # this cannot be skipped since a reward function takes numpy arrays as its inputs
            states = next_s.detach().numpy()

        best_seq = np.argmax(scores)

        return action_sequences[0, best_seq]
