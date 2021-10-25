from learner import Learner
from discrete_pendulum import DiscretePendulumEnv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

gamma = 0.99
env = DiscretePendulumEnv()
learner = Learner(num_states=env.observation_space.n,
                  num_actions=env.action_space.n,
                  gamma=gamma,
                  )

q_opt = np.load('optimum.npy')
q_learned = np.load('table.npy')
v_opt = np.max(q_opt[1250])
v_learned = np.max(q_learned[1250])


horizon = np.arange(0, 10000000+500000, 500000)
plt.plot(horizon, np.abs(v - v_star), color='purple', linestyle='dashed')
plt.title('Q-learning: Discrete Pendulum')
plt.xlim(0, horizon[-1])
plt.xlabel(r'$t$')
plt.ylabel(r'$\vert v_t(s_{1250}) -v^\star(s_{1250}) \vert$')
plt.ylim(0)
plt.grid()
plt.savefig('q_learning_progress_at_s=1250.pdf')
