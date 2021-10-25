import time
import csv
import copy
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Independent
from torch.distributions.normal import Normal
from torch.optim import Adam
from itertools import chain
from memory import OnPolicyMemory
from utils import *


def ppo_train(env, agent, max_iter,
              gamma=0.99, lr=3e-4, lam=0.95, delta=1e-3,
              epsilon=0.2,
              steps_per_epoch=10000, eval_interval=10000,
              snapshot_interval=10000, device='cpu'
              ):

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    max_ep_len = env._max_episode_steps
    memory = OnPolicyMemory(obs_dim, act_dim, gamma, lam, lim=steps_per_epoch)
    test_env = copy.deepcopy(env)

    params = chain(agent.pi.parameters(), agent.V.parameters())
    optimizer = Adam(params, lr=lr)
    save_path = './ppo_snapshots/'
    os.makedirs(save_path, exist_ok=True)
    os.makedirs('./ppo_learning_curves/', exist_ok=True)
    log_file = open('./ppo_learning_curves/res.csv',
                    'w',
                    encoding='utf-8',
                    newline=''
                    )
    logger = csv.writer(log_file)
    num_epochs = max_iter // steps_per_epoch
    total_t = 0
    begin = time.time()
    for epoch in range(num_epochs):
        # start agent-env interaction
        state = env.reset()
        step_count = 0
        ep_reward = 0

        for t in range(steps_per_epoch):
            # collect transition samples by executing the policy
            action, log_prob, v = agent.act(state)

            next_state, reward, done, _ = env.step(action)
            memory.append(state, action, reward, v, log_prob)

            ep_reward += reward
            step_count += 1

            if (step_count == max_ep_len) or (t == steps_per_epoch - 1):
                # termination of env by env wrapper, or by truncation due to memory size
                s_last = torch.tensor(next_state, dtype=torch.float).to(device)
                v_last = agent.V(s_last).item()
                memory.compute_values(v_last)
            elif done:
                # episode done as the agent reach a terminal state
                v_last = 0.0
                memory.compute_values(v_last)

            state = next_state

            if done:
                state = env.reset()
                step_count = 0
                ep_reward = 0

            if total_t % eval_interval == 0:
                avg_score, std_score = evaluate(agent, test_env, num_episodes=5)
                elapsed_t = time.time() - begin
                print('[elapsed time : {:.1f}s| iter {}] score = {:.2f}'.format(elapsed_t, total_t, avg_score),
                      u'\u00B1', '{:.4f}'.format(std_score))
                evaluation_log = [t, avg_score, std_score]
                logger.writerow(evaluation_log)

            if total_t % snapshot_interval == 0:
                snapshot_path = save_path + 'iter{}_'.format(total_t)
                # save weight & training progress
                save_snapshot(agent, snapshot_path)

            total_t += 1

        # train agent at the end of each epoch
        ppo_update(agent, memory, optimizer, epsilon, num_updates=1)
    log_file.close()
    return


def ppo_update(agent, memory, optimizer, epsilon, num_updates=1, device='cpu'):

    batch = memory.load()
    states = torch.Tensor(batch['state']).to(device)
    actions = torch.Tensor(batch['action']).to(device)
    target_v = torch.Tensor(batch['val']).to(device)
    A = torch.Tensor(batch['A']).to(device)
    old_log_probs = torch.Tensor(batch['log_prob']).to(device)

    for _ in range(num_updates):
        ################
        # train critic #
        ################
        log_probs, ent = agent.pi.compute_log_prob(states, actions)

        # compute prob ratio
        # $\frac{\pi(a_t | s_t ; \theta)}{\pi(a_t | s_t ; \theta_\text{old})}$
        r = torch.exp(log_probs - old_log_probs)
        # construct clipped loss
        # $r^\text{clipped}_t(\theta) = \text{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon)$
        clipped_r = torch.clamp(r, 1 - epsilon, 1 + epsilon)
        # surrogate objective for each $t$
        # $\min \{ r_t(\theta) \hat{A}_t, r^\text{clipped}_t(\theta) \hat{A}_t \}$
        single_step_obj = torch.min(r * A, clipped_r * A)
        pi_loss = -torch.mean(single_step_obj)

        v = agent.V(states)
        V_loss = torch.mean((v - target_v) ** 2)
        ent_bonus = torch.mean(ent)

        loss = pi_loss + 0.5 * V_loss - 0.01 * ent_bonus
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return


class PPOAgent:
    def __init__(
                 self,
                 dimS,
                 dimA,
                 hidden1=64,
                 hidden2=32,
                 device='cpu'
                 ):

        self.dimS = dimS
        self.dimA = dimA
        self.device = device

        self.pi = StochasticPolicy(dimS, dimA, hidden1, hidden2).to(device)
        self.V = ValueFunction(dimS, hidden1, hidden2).to(device)

    def act(self, state):
        state = torch.tensor(state, dtype=torch.float).to(self.device)
        with torch.no_grad():
            action, log_prob = self.pi(state)
            val = self.V(state)

        action = action.cpu().detach().numpy()
        log_prob = log_prob.cpu().detach().numpy()
        val = val.cpu().detach().numpy()

        return action, log_prob, val


class StochasticPolicy(nn.Module):
    def __init__(self, dimS, dimA, hidden1, hidden2):
        super(StochasticPolicy, self).__init__()
        self.fc1 = nn.Linear(dimS, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, dimA)

        # log_sigma as learnable params
        log_sigma = -0.5 * torch.ones(dimA, requires_grad=True)
        self.log_sigma = torch.nn.Parameter(log_sigma)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        mu = self.fc3(x)
        sigma = torch.exp(self.log_sigma)

        # sample an action from multivariate normal distribution with diagonal covariance
        distribution = Independent(Normal(mu, sigma), 1)
        action = distribution.rsample()

        log_prob = distribution.log_prob(action)

        return action, log_prob

    def compute_log_prob(self, state, action):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        mu = self.fc3(x)
        sigma = torch.exp(self.log_sigma)
        distribution = Independent(Normal(mu, sigma), 1)
        # log probability $\log \pi(a | s ; \theta)$
        log_prob = distribution.log_prob(action)
        # entropy of the current policy $\mathbb{H}(\pi(\cdot | s ; \theta))$
        ent = distribution.entropy()
        return log_prob, ent


class ValueFunction(nn.Module):
    # state value function
    def __init__(self, dimS, hidden1, hidden2):
        super(ValueFunction, self).__init__()
        self.fc1 = nn.Linear(dimS, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, 1)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


def evaluate(agent, env, num_episodes=5):

    scores = np.zeros(num_episodes)
    for i in range(num_episodes):
        obs = env.reset()
        done = False
        score = 0.
        while not done:
            action = agent.act(obs)[0]
            obs, rew, done, _ = env.step(action)
            score += rew

        scores[i] = score
    avg_score = np.mean(scores)
    std_score = np.std(scores)
    return avg_score, std_score