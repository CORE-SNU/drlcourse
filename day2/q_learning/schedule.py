class VisitCountStepsizeSchedule:
    def __init__(self, deg=1.0):
        # polynomial stepsize schedule : $\Theta(N_t(s, a)^{-d})$
        # $N_t(s, a)$ is the number of visits of (s, a)-pair until step t
        # to satisfy Robbins-Monro condition, d must satisfy $d \in (1/2, 1]$
        assert .5 < deg <= 1
        self.deg = deg

    def __call__(self, n):
        return 1. / ((n + 1.) ** self.deg)


class LinearExplorationSchedule:
    def __init__(self, rollout_len, initial_epsilon=1., final_epsilon=0.02):
        # linear exploration schedule
        self.decrement = (initial_epsilon - final_epsilon) / rollout_len
        self.initial_epsilon = initial_epsilon
        self.final_epsilon = final_epsilon

    def __call__(self, t):
        # define this as callable object so that the schedule is stateless
        return max(self.initial_epsilon - t * self.decrement, self.final_epsilon)
