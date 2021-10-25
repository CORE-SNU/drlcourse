import numpy as np
from numpy import arctan
from my_pendulum import angle_normalize


def pendulum_reward(states, u):
    # vectorized version of reward function of Pendulum-v0 task
    # the number of given states and actions must coincide
    # output a numpy array of shape (K,), where K is the number of given state-action pair

    u = np.squeeze(u)   # remove extra dimension, otherwise its shape would be (K, 1) instead of (K, )
    th, th_dot = states[:, 0], states[:, 1]
    costs = angle_normalize(th) ** 2 + .1 * th_dot ** 2 + .001 * (u ** 2)
    return -costs


def cheetah_reward(states, action):
    vel = states[:, 3]
    return vel + np.sum(action ** 2, axis=1)
    
