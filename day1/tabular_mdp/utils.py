import numpy as np


def q_ftn(P, R, gamma, v):
    """
    given v, get corresponding q
    """
    return R + gamma * np.reshape(np.matmul(P, v), newshape=R.shape, order='F')


def induced_dynamic(nS, P, R, pi):
    """
    given policy pi, compute induced dynamic P^pi & R^pi
    """
    S = range(nS)
    # P_pi = np.array([P[pi[s] * nS + s, :] for s in S])
    rows = np.arange(nS) + nS * pi
    P_pi = P[rows]
    R_pi = np.array([[R[s, pi[s]]] for s in range(nS)])

    return P_pi, R_pi


def eval_policy(nS, P, R, gamma, pi):
    """
    policy evaluation
    """
    P_pi, R_pi = induced_dynamic(nS, P, R, pi)

    Id = np.identity(nS)

    # discounted reward problem
    v_pi = np.linalg.solve(Id - gamma * P_pi, R_pi)
    return v_pi



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
    v_next = np.max(q, axis=1, keepdims=True)   # computation of Bellman operator Tv

    return v_next


def print_result(v, pi, mode='discount'):
    print('+========== Result ==========+')
    nS = v.size
    S = range(nS)
    
    print('optimal value function : ')
    # print a given value function
    for s in S:
        print('v(s{}) = {}'.format(s + 1, v[s, 0]))

    print('optimal policy : ')
    # print a given policy
    for s in S:
        print('pi(s{}) = a{}'.format(s + 1, pi[s] + 1))
    return
