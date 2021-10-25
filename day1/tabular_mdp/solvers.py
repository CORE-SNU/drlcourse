import numpy as np
from scipy.optimize import linprog

from utils import *

"""
each solver outputs the optimal value function v, which is a numpy matrix of shape (|S|, 1), 
and an optimal policy, a numpy array of shape (|S|,)
"""


def VI(P, R, gamma):
    """
    implementation of value iteration
    """
    EPS = 1e-6
    nS, nA = R.shape
    # initialize v
    v = np.zeros(shape=(nS, 1), dtype=np.float)

    while True:
        v_next = bellman_update(P, R, gamma, v)
        if np.linalg.norm(v_next - v, ord=np.inf) < EPS:
            break
        v = v_next

    pi = greedy(P, R, gamma, v)

    return v, pi


def PI(P, R, gamma):
    """
    implementation of policy iteration
    """
    nS, nA = R.shape

    # initialize policy
    pi = np.random.randint(nA, size=nS)

    while True:
        v = eval_policy(nS, P, R, gamma, pi)
        pi_next = greedy(P, R, gamma, v)
        if (pi_next == pi).all():
            break
        pi = pi_next

    return v, pi


def LP(P, R, gamma):
    
    nS, nA = R.shape
    Id = np.tile(np.identity(nS), reps=(nA, 1))
    A = gamma * P - Id
    b = -np.reshape(R, newshape=(nS * nA, 1), order='F')
    c = 1.0 * np.ones(nS)

    res = linprog(c, A, b, bounds=(None, None))

    v = np.reshape(res['x'], newshape=(nS, 1))
    pi = pi = greedy(P, R, gamma, v)

    return v, pi
