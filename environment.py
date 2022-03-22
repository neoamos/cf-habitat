from typing import Optional, Type

import numpy as np

import habitat
from habitat import Config, Dataset
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.environments import NavRLEnv

@baseline_registry.register_env(name="CFNavRLEnv")
class CFNavRLEnv(NavRLEnv):
    def __init__(self, config: Config, dataset: Optional[Dataset] = None):
        super().__init__(config, dataset)

    def reset(self):
        # Reset env until we get a valid episode.
        while True:
            observations = super().reset()
            self._previous_action = None
            self._previous_measure = self._env.get_metrics()[
                self._reward_measure_name
            ]
            current_position = self._env._sim.get_agent_state().position
            goal = self._env.current_episode.goals[0].position
            geodesic_distance = self._env._sim.geodesic_distance(current_position, goal)
            if geodesic_distance != np.inf and geodesic_distance > 0.0:
                return observations

    def get_reward(self, observations):
        reward = self._rl_config.SLACK_REWARD

        current_measure = self._env.get_metrics()[self._reward_measure_name]

        reward += self._previous_measure - current_measure
        self._previous_measure = current_measure

        if self._episode_success():
            reward += self._rl_config.SUCCESS_REWARD

        if hasattr(self._rl_config, 'COLLISION_REWARD') and self._env._sim.previous_step_collided:
            reward += self._rl_config.COLLISION_REWARD
            
        return reward