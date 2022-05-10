import copy
import logging

import gym
import numpy as np
from PIL import ImageColor
from gym import spaces
from gym.utils import seeding

from ..utils.action_space import MultiAgentActionSpace
from ..utils.draw import draw_grid, fill_cell, draw_circle, write_cell_text
from ..utils.observation_space import MultiAgentObservationSpace

logger = logging.getLogger(__name__)


"""
Matrix pay-off game implemented to be integrated with the collection of environments MA-gym
"""

class Matrix_Env_2(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, n_agents=2):
        self.n_agents = n_agents
        self._max_steps = 1
        self._step_count = 0

        self.action_space = MultiAgentActionSpace([spaces.Discrete(3) for _ in range(self.n_agents)])

        self.payoff_matrix = np.array([
            [8, -12, -12],
            [-12, 0, 0],
            [-12, 0, 0]
        ])


        print(self.action_space[0].n)

        self.state = np.ones(5)

    def get_avail_agent_actions(self, agent_id):
        return [1 for _ in range(3)]

    def action_space_sample(self):
        return [agent_action_space.sample() for agent_action_space in self.action_space]

    def get_state(self):
        return self.state

    def get_obs(self):
        return [self.state for _ in range(self.n_agents)]

    def reset(self):
        self._step_count = 0
        return self.get_obs()

    def step(self, agents_action):
        reward = self.payoff_matrix[agents_action[0], agents_action[1]]
        terminated = True

        return reward, terminated, {}


