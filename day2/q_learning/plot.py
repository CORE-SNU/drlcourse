from table import QTable
from discrete_pendulum import DiscretePendulumEnv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


env = DiscretePendulumEnv()
learner = QTable(num_states=env.observation_space.n,
                 num_actions=env.action_space.n,
                 pth='./'
                 )

V = np.max(learner.Q, axis=1)
landscape = np.reshape(V, newshape=(env.n_thdot + 1, env.n_th), order='F')
extended = np.zeros((env.n_thdot + 1, env.n_th + 1))
extended[:, :env.n_th] = landscape
extended[:, env.n_th] = landscape[:, 0]

th = np.arange(-np.pi, np.pi + env.width_th, env.width_th)
thdot = np.arange(-env.max_speed, env.max_speed + env.width_thdot, env.width_thdot)

th, thdot = np.meshgrid(th, thdot)

plt.contourf(th, thdot, extended, cmap='RdGy')
plt.colorbar()
plt.xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi], [r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'])
plt.yticks([-8, -4, 0, 4, 8])
plt.title('Pendulum: Q-learning')
plt.xlabel(r'$\theta (rad)$')
plt.ylabel(r'$\dot\theta (rad/s)$')
plt.savefig('learned_value.png')
