import torch
from torch.optim import Adam
import numpy as np
import gym
import copy

from buffer import ReplayBuffer
from sac_model import SACActor, DoubleCritic
from utils import freeze, unfreeze


class SACAgent:
    def __init__(self,
                 dimS,
                 dimA,
                 ctrl_range,
                 gamma=0.99,
                 pi_lr=1e-4,
                 q_lr=1e-3,
                 polyak=1e-3,
                 alpha=0.2,
                 hidden1=400,
                 hidden2=300,
                 buffer_size=1000000,
                 batch_size=128,
                 device='cpu',
                 render=False):

        self.dimS = dimS
        self.dimA = dimA
        self.ctrl_range = ctrl_range

        self.gamma = gamma
        self.pi_lr = pi_lr
        self.q_lr = q_lr
        self.polyak = polyak
        self.alpha = alpha

        self.batch_size = batch_size
        # networks definition
        # pi : actor network, Q : 2 critic network
        self.pi = SACActor(dimS, dimA, hidden1, hidden2, ctrl_range).to(device)
        self.Q = DoubleCritic(dimS, dimA, hidden1, hidden2).to(device)

        # target networks
        self.target_Q = copy.deepcopy(self.Q).to(device)

        freeze(self.target_Q)

        self.buffer = ReplayBuffer(dimS, dimA, limit=buffer_size)

        self.Q_optimizer = Adam(self.Q.parameters(), lr=self.q_lr)
        self.pi_optimizer = Adam(self.pi.parameters(), lr=self.pi_lr)

        self.device = device
        self.render = render

        return

    def act(self, state, eval=False):

        state = torch.tensor(state, dtype=torch.float).to(self.device)
        with torch.no_grad():
            action, _ = self.pi(state, eval=eval, with_log_prob=False)
        action = action.cpu().detach().numpy()

        return action

    def target_update(self):

        for params, target_params in zip(self.Q.parameters(), self.target_Q.parameters()):
            target_params.data.copy_(self.polyak * params.data + (1.0 - self.polyak) * target_params.data)

        return

    def train(self):

        device = self.device

        batch = self.buffer.sample_batch(batch_size=self.batch_size)


        # unroll batch
        obs_batch = torch.tensor(batch.obs, dtype=torch.float).to(device)
        act_batch = torch.tensor(batch.act, dtype=torch.float).to(device)
        next_obs_batch = torch.tensor(batch.next_obs, dtype=torch.float).to(device)
        rew_batch = torch.tensor(batch.rew, dtype=torch.float).to(device)
        done_batch = torch.tensor(batch.done, dtype=torch.float).to(device)

        masks = torch.tensor([1.]) - done_batch

        with torch.no_grad():
            next_actions, log_probs = self.pi(next_obs_batch, with_log_prob=True)
            target_q1, target_q2 = self.target_Q(next_obs_batch, next_actions)
            target_q = torch.min(target_q1, target_q2)
            target = rew_batch + self.gamma * masks * (target_q - self.alpha * log_probs)

        out1, out2 = self.Q(obs_batch, act_batch)

        Q_loss1 = torch.mean((out1 - target)**2)
        Q_loss2 = torch.mean((out2 - target)**2)
        Q_loss = Q_loss1 + Q_loss2

        self.Q_optimizer.zero_grad()
        Q_loss.backward()
        self.Q_optimizer.step()

        actions, log_probs = self.pi(obs_batch, with_log_prob=True)

        freeze(self.Q)
        q1, q2 = self.Q(obs_batch, actions)
        q = torch.min(q1, q2)

        pi_loss = torch.mean(self.alpha * log_probs - q)

        self.pi_optimizer.zero_grad()
        pi_loss.backward()
        self.pi_optimizer.step()

        unfreeze(self.Q)

        self.target_update()

        return

    def single_eval(self, env_id, render=False):
        """
        evaluation of the agent on a single episode
        """
        env = gym.make(env_id)
        state = env.reset()
        ep_reward = 0
        done = False

        while not done:
            if render:
                env.render()

            action = self.act(state, eval=True)
            state, reward, done, _ = env.step(action)

            ep_reward += reward
        if render:
            env.close()

        return ep_reward

    def eval(self, env_id, t, eval_num=10):

        scores = np.zeros(eval_num)

        for i in range(eval_num):
            render = True if (self.render == True and i == 0) else False
            scores[i] = self.single_eval(env_id, False)

        avg = np.mean(scores)

        print('step {} : {:.4f}'.format(t,  avg))

        return [t, avg]

    def save_model(self, path):
        print('adding checkpoints...')
        checkpoint_path = path + 'model.pth.tar'
        torch.save(
                    {'actor': self.pi.state_dict(),
                     'critic': self.Q.state_dict(),
                     'target_critic': self.target_Q.state_dict(),
                     'actor_optimizer': self.pi_optimizer.state_dict(),
                     'critic_optimizer': self.Q_optimizer.state_dict()
                    },
                    checkpoint_path)

        return

    def load_model(self, path):
        print('networks loading...')
        checkpoint = torch.load(path)

        self.pi.load_state_dict(checkpoint['actor'])
        self.Q.load_state_dict(checkpoint['critic'])
        self.target_Q.load_state_dict(checkpoint['target_critic'])
        self.pi_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.Q_optimizer.load_state_dict(checkpoint['critic_optimizer'])

        return