import numpy as np


class OnPolicyMemory:
    def __init__(self,
                 dimS,
                 dimA,
                 gamma,
                 lam,
                 lim
                 ):

        self._obs_mem = np.zeros(shape=(lim, dimS))
        self._act_mem = np.zeros(shape=(lim, dimA))
        self._rew_mem = np.zeros(shape=(lim,))
        self._val_mem = np.zeros(shape=(lim,))
        self._log_prob_mem = np.zeros(shape=(lim,))

        # memory of cumulative rewards which are MC-estimates of the current value function
        self._target_v_mem = np.zeros(shape=(lim,))
        # memory of GAE($\lambda$)-estimate of the current advantage function
        self._adv_mem = np.zeros(shape=(lim,))

        self._gamma = gamma
        self._lam = lam

        self._lim = lim    # current size of the memory
        self._size = 0
        self._ep_start = 0
        self._head = 0       # position to save next transition sample

    def append(self, state, action, reward, value, log_prob):

        assert self._head < self._lim
        self._obs_mem[self._head, :] = state
        self._act_mem[self._head, :] = action
        self._rew_mem[self._head] = reward
        self._val_mem[self._head] = value
        self._log_prob_mem[self._head] = log_prob

        self._head += 1
        self._size += 1

        return

    def load(self):
        # load samples when the memory is full
        assert self._size == self._lim

        states = self._obs_mem[:]
        actions = self._act_mem[:]
        target_v = self._target_v_mem[:]
        GAEs = self._adv_mem[:]
        log_probs = self._log_prob_mem[:]

        # apply advantage normalization trick
        GAEs = (GAEs - np.mean(GAEs)) / np.std(GAEs)

        batch = {'state': states,
                 'action': actions,
                 'val': target_v,
                 'A': GAEs,
                 'log_prob': log_probs}

        self._size, self._ep_start, self._head = 0, 0, 0

        return batch

    def compute_values(self, v_last):
        # compute advantage estimates & target values at the end of each episode
        # $v = 0$ if $s_T$ is terminal, else $v = \hat{V}(s_T)$
        gamma = self._gamma
        lam = self._lam
        start = self._ep_start   # 1st step of the epi, corresponds to t = 0
        idx = self._head - 1      # last step of the epi, corresponds to t = T - 1

        v = np.zeros(idx - start + 2)
        v[-1] = v_last
        v[:-1] = self._val_mem[start: idx + 1]

        # compute TD-error based on value estimate
        # $\hat{\delta}_t = r_t + \gamma \hat{V}(s_{t+1}) - \hat{V}(s_t)$, t < T
        delta = self._rew_mem[start: idx + 1] + gamma * v[1:] - v[:-1]

        # backward calculation of cumulative rewards & GAE
        next_GAE = 0.0
        # for a truncated episode, the last reward is set to 0
        # otherwise, the last reward is set to $\hat{V}(s_T)$
        next_R = v_last

        # $R_T = 0$ if $s_T$ is terminal, else $R_T = \hat{V}(s_T)$, $\hat{A}_T = 0$
        # for t = T-1, ..., 0:
        # do
        #     $\hat{A}_t = \delta_t + \gamma (\lambda \delta)_{t+1}$,
        #     $R_t = \r_t + \gamma R_{t+1}$
        # done

        for t in range(idx, start - 1, -1):
            self._target_v_mem[t] = self._rew_mem[t] + gamma * next_R
            self._adv_mem[t] = delta[t - start] + (gamma * lam) * next_GAE

            next_R = self._target_v_mem[t]
            next_GAE = self._adv_mem[t]

        self._ep_start = self._head

        return

    def sample_batch(self, batch_size):
        """
        sample a mini-batch from the buffer
        :param batch_size: size of a mini-batch to sample
        :return: mini-batch of transition samples
        """

        rng = np.random.default_rng()
        idxs = rng.choice(self._lim, batch_size)

        states = self._obs_mem[idxs, :]
        actions = self._act_mem[idxs, :]
        target_vals = self._target_v_mem[idxs]
        GAEs = self._adv_mem[idxs]
        log_probs = self._log_prob_mem[idxs]

        batch = {'state': states,
                 'action': actions,
                 'val': target_vals,
                 'A': GAEs,
                 'log_prob': log_probs
        }

        return batch
