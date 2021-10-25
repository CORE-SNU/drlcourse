import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path
from typing import List


class DiscretePendulumEnv(gym.Env):
    # forked from https://github.com/openai/gym/blob/master/gym/envs/classic_control/pendulum.py
    # modified so that the environment eventually becomes a discrete MDP with stochastic transition
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, g=10.0, n_th=40, n_thdot=60, num_u=15):
        # system parameters
        self.max_speed = 8
        self.max_torque = 2.
        self.dt = .05
        self.g = g
        self.m = 1.
        self.L = 1.

        self.viewer = None      # for rendering

        # note that (# of points along \dot\theta-axis) is (# of cells + 1)!
        # however, (# of points along \theta-axis) = (# of cells)
        self.n_th = n_th        # number of cells along \theta-axis
        self.n_thdot = n_thdot     # number of cells along \dot\theta-axis

        self.n_pts = self.n_th * (self.n_thdot + 1)
        self.width_th = 2. * np.pi / self.n_th
        self.width_thdot = 2. * self.max_speed / self.n_thdot
        self.num_u = num_u
        self.u_interval = 2. * self.max_torque / (self.num_u - 1)
        self.action_space = spaces.Discrete(self.num_u)
        self.observation_space = spaces.Discrete(self.n_pts)

        self.x = None
        self.seed()

    def step(self, action: int):
        assert action < self.action_space.n
        u = -self.max_torque + action * self.u_interval     # discrete to continuous
        costs = self.cost(self.x, u)
        # s_{t+1} ~ p(s_t, a_t)
        x_next = self.f(x=self.x, u=u)
        discrete_state = self.to_discrete(x_next)
        self.x = self.to_continuous(discrete_state)
        return discrete_state, -costs, False, {}

    def reset(self, deterministic=False):
        if deterministic:
            discrete_state = 30
        else:
            high = np.array([np.pi, 1])
            discrete_state = self.to_discrete(self.np_random.uniform(low=-high, high=high))
        self.x = self.to_continuous(discrete_state)
        self.last_u = None
        return discrete_state

    def f(self, x, u):
        # simulate pendulum dynamics in continuous state space
        th, thdot = x
        g = self.g
        m = self.m
        L = self.L
        dt = self.dt
        newthdot = thdot + (-3. * g / (2. * L) * np.sin(th + np.pi) + 3. / (m * L ** 2) * u) * dt
        newth = angle_normalize(th + newthdot * dt)
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)
        x_next = np.array([newth, newthdot])
        return x_next

    @staticmethod
    def cost(x, u):
        # quadratic cost ftn
        th, thdot = x
        return th ** 2 + .1 * thdot ** 2 + .001 * (u ** 2)

    def to_continuous(self, d) -> np.ndarray:
        n1, n2 = d // (self.n_thdot + 1), d % (self.n_thdot + 1)
        x1, x2 = -np.pi + n1 * self.width_th, -self.max_speed + n2 * self.width_thdot
        return np.array([x1, x2])

    def to_discrete(self, x) -> int:
        # sample a discrete state corresponding to the current continuous state
        # computed under induced stochastic transition
        p = self.bin(x)
        d = np.random.choice(self.n_pts, p=p)
        return d

    def bin(self, x: np.ndarray) -> List[float]:
        # binning continuous state variable using Kuhn triangulation
        # easily generalizes to high-dimensional case
        # return a probability vector $p$, where $p(i)$ indicates the probability of $x$ being approximated by a pt $i$
        th, thdot = x
        n1, x1 = int((th + np.pi) // self.width_th), (th + np.pi) % self.width_th
        n2, x2 = int((thdot + self.max_speed) // self.width_thdot), (thdot + self.max_speed) % self.width_thdot
        # normalize both x1 and x2 so that $x = (x1, x2)^\top$ belongs to the unit cube $[0, 1)^2$
        x1 /= self.width_th
        x2 /= self.width_thdot
        d = n1 * (self.n_thdot + 1) + n2
        # subdivision of a cube into two simplices (n! in general)
        # mesh arranged in lexicographic manner as follows:
        # $x_1 > x_2$ iff $\theta_1 > theta_2$ or $\theta_1 = \theta_2, \dot\theta_1 > \dot\theta_2$
        p = [0.] * self.n_pts
        if x1 >= x2:
            # since the original state space is cylindrical, take mod N, where N = (size of the state space)
            d1, d2, d3 = d, (d + self.n_thdot + 1) % self.n_pts, (d + 2 + self.n_thdot) % self.n_pts
            p[d1], p[d2], p[d3] = 1. - x1, x1 - x2, x2
        else:
            d1, d2, d3 = d, (d + 1) % self.n_pts, (d + 2 + self.n_thdot) % self.n_pts
            p[d1], p[d2], p[d3] = 1 - x2, x2 - x1, x1
        return p

    def render(self, mode='human'):
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-2.2, 2.2, -2.2, 2.2)
            rod = rendering.make_capsule(1, .2)
            rod.set_color(.8, .3, .3)
            self.pole_transform = rendering.Transform()
            rod.add_attr(self.pole_transform)
            self.viewer.add_geom(rod)
            axle = rendering.make_circle(.05)
            axle.set_color(0, 0, 0)
            self.viewer.add_geom(axle)
            fname = path.join(path.dirname(__file__), "assets/clockwise.png")
            self.img = rendering.Image(fname, 1., 1.)
            self.imgtrans = rendering.Transform()
            self.img.add_attr(self.imgtrans)

        self.viewer.add_onetime(self.img)
        self.pole_transform.set_rotation(self.x[0] + np.pi / 2)
        if self.last_u:
            self.imgtrans.scale = (-self.last_u / 2, np.abs(self.last_u) / 2)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


def angle_normalize(theta):
    # $[-pi, pi)$
    return ((theta + np.pi) % (2. * np.pi)) - np.pi

