import numpy as np
import matplotlib.pyplot as plt


class GridWorld:
    length = 7
    assert length >= 4
    n = length**2  # cardinality of state space
    m = 4   # cardinality of action space

    # transition & reward matrix construction
    P = np.zeros((n*m, n))
    R = np.zeros((n, m))

    SPAWN = (length - 4) * length
    GOAL = length - 2
    TELEPORT = (length-1)*length+4

    for i in range(n):
        # exclude special states in advance
        if i not in [TELEPORT, GOAL]:
            # action = 0 (MOVE UP)
            if i < length:
                # hit wall
                P[i, i] = 1.
                R[i, 0] = -1.
            else:
                P[i, i-length] = 1.
            # action = 1 (MOVE LEFT)
            if i % length == 0:
                P[i+n, i] = 1.
                R[i, 1] = -1.
            else:
                P[i+n, i-1] = 1.
            # action = 2 (MOVE RIGHT)
            if i % length == length - 1:
                P[i+2*n, i] = 1.
                R[i, 2] = -1.
            else:
                P[i+2*n, i+1] = 1.
            # action = 3 (MOVE DOWN)
            if i >= (length-1)*length:
                P[i+3*n, i] = 1.
                R[i, 3] = -1.
            else:
                P[i+3*n, i+length] = 1.

    for k in range(m):
        # handle special states
        P[TELEPORT+n*k, GOAL] = 1.
        P[GOAL+n*k, SPAWN] = 1.
        R[GOAL, k] = 10

    def __init__(self):
        return


def plot_heatmap(v, w):
    value_map_v = np.reshape(v, newshape=(7, 7))
    value_map_w = np.reshape(w, newshape=(7, 7))
    fig, ax = plt.subplots(1, 2, figsize=(10, 8))
    
    im_v = ax[0].imshow(value_map_v, cmap='cividis')
    im_w = ax[1].imshow(value_map_w, cmap='cividis')
    ax[0].axis('off')
    ax[1].axis('off')
    ax[0].set_xticklabels([])
    ax[0].set_yticklabels([])
    ax[1].set_xticklabels([])
    ax[1].set_yticklabels([])

    for i in range(7):
        for j in range(7):
            ax[0].text(j, i, '{:.2f}'.format(value_map_v[i, j]), ha='center', va='center', color='w')
            ax[1].text(j, i, '{:.2f}'.format(value_map_w[i, j]), ha='center', va='center', color='w')
    ax[0].set_title("GridWorld : Value Iteration")
    ax[1].set_title("GridWorld : Policy Iteration")
    plt.tight_layout()

