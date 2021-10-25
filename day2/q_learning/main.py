import numpy as np
from tqdm import tqdm
from table import QTable
from schedule import VisitCountStepsizeSchedule, LinearExplorationSchedule
from discrete_pendulum import DiscretePendulumEnv


env = DiscretePendulumEnv()
num_states = env.observation_space.n
num_actions = env.action_space.n
gamma = 0.99

learner = QTable(num_states=env.observation_space.n, num_actions=env.action_space.n, gamma=gamma)
rollout_len = 1000000
visit_count = np.zeros(shape=(num_states, num_actions))     # save visit counts N(s, a) of all state-action pairs
alpha = VisitCountStepsizeSchedule(deg=0.5001)
epsilon = LinearExplorationSchedule(rollout_len, final_epsilon=0.4)


checkpoint_interval = rollout_len // 20

s = env.reset()
for t in tqdm(range(rollout_len + 1)):
    u = np.random.rand()
    if u < epsilon(t):
        a = env.action_space.sample()
    else:
        a = learner.act(state=s)

    s_next, r, _, _ = env.step(action=a)
    # env.render()
    n = visit_count[s, a]
    learner.update(state=s, action=a, reward=r, next_state=s_next, alpha=alpha(n))
    visit_count[s, a] += 1
    s = s_next
    if t % checkpoint_interval == 0:
        learner.save()

