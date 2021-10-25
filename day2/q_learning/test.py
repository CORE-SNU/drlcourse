import numpy as np
import matplotlib.pyplot as plt
from table import QTable
from discrete_pendulum import DiscretePendulumEnv


gamma = 0.99
env = DiscretePendulumEnv()
learner = QTable(num_states=env.observation_space.n,
                 num_actions=env.action_space.n,
                 pth='./'
                 )

ep_len = 400
# test learned result!
trajectory = np.zeros((ep_len, 2))      # store continuous states
reward = 0.
s = env.reset()

for t in range(ep_len):
    trajectory[t] = np.copy(env.x)
    a = learner.act(s)
    s, r, _, _ = env.step(a)
    reward += (gamma ** t) * r
    env.render()
print('total reward =', reward)
env.close()

fig, ax = plt.subplots(2, 1, figsize=(8, 10), sharex=True)
x = env.dt * np.arange(ep_len)
ylabels = [r'$\theta$ (rad)', r'$\dot\theta$ (rad/s)']
ax[1].set_xlabel(r'$t$ (s)', fontsize=20)
ax[0].set_ylim(-np.pi, np.pi)
ax[1].set_ylim(-8., 8.)
ax[0].set_yticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
ax[0].set_yticklabels([r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'])
ax[1].set_yticks([-8, -4, 0, 4, 8])
for i in range(2):
    ax[i].plot(x, trajectory[:, i])
    ax[i].set_xlim(0, x[-1])
    ax[i].grid(True)
    ax[i].set_ylabel(ylabels[i], fontsize=20)
    ax[i].tick_params(axis='both', which='major', labelsize=18)
fig.tight_layout()
fig.savefig('trajectory.png')
