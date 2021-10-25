import numpy as np


def value_iteration(P, R):
    nS, nA = R.shape
    h = np.zeros(shape=(nS, 1), dtype=np.float)
    next_h = bellman_update(P, R, 1.0, h)
    next_h -= next_h[-1, 0]
    while np.linalg.norm(next_h - h, ord=np.inf) > 1e-6: 
        h = next_h
        next_h = bellman_update(P, R, 1.0, h)
        next_h -= next_h[-1, 0]

    gain = (bellman_update(P, R, 1.0, h) - h)[-1, 0]
    pi = greedy(P, R, 1.0, h)

    return gain, pi


def q_ftn(P, R, gamma, v):
    """
    given v, get corresponding q
    """
    return R + gamma * np.reshape(np.matmul(P, v), newshape=R.shape, order='F')


def greedy(P, R, gamma, v):
    """
    construct greedy policy by pi(s) = argmax_a q(s, a)
    """
    q = q_ftn(P, R, gamma, v)
    pi = np.argmax(q, axis=1)

    return pi


def bellman_update(P, R, gamma, v):
    """
    implementation of one-step Bellman update
    return : vector of shape (|S|, 1) which corresponds to Tv, where T is Bellman operator
    """

    q = q_ftn(P, R, gamma, v)
    v_next = np.max(q, axis=1, keepdims=True)

    return v_next
