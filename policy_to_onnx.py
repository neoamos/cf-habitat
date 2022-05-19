import torch
import argparse
import numpy as np

import policy
import observation_transformers

import habitat
from habitat_baselines.config.default import get_config as get_baselines_config
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.utils.common import (
    batch_obs,
)

from gym import spaces
from habitat.core.spaces import ActionSpace, EmptySpace

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp-config",
        type=str,
        required=True,
        help="path to config yaml containing info about policy",
    )

    args = parser.parse_args()

    config = get_baselines_config(args.exp_config)

    task_config = config.TASK_CONFIG
    possible_actions = task_config.TASK.POSSIBLE_ACTIONS
    width = 256 #config.SIMULATOR.RGB_SENSOR.WIDTH
    height = 256 #config.SIMULATOR.RGB_SENSOR.HEIGHT

    observation_space = spaces.Dict({
      "pointgoal_with_gps_compass": spaces.Box(
        low=-3.4028235e+38, high=3.4028235e+38, shape=(2,)
        ),
      "rgb": spaces.Box(low=0, high=255, shape=(width, height, 3), dtype=np.uint8)
    })
    action_space = ActionSpace(
      {a : EmptySpace() for a in possible_actions}
    )
    action_shape = (1,)
    action_type = torch.long

    policy = baseline_registry.get_policy(config.RL.POLICY.name)
    actor_critic = policy.from_config(
      config, observation_space, action_space
    )

    print(actor_critic)
    with torch.no_grad():
        hidden_state = torch.zeros(
            1,
            actor_critic.net.num_recurrent_layers,
            config.RL.PPO.hidden_size,
            device="cpu",
        )
        prev_action = torch.zeros(
            1,
            *action_shape,
            device="cpu",
            dtype=action_type,
        )
        not_done_masks = torch.tensor(
            [True],
            dtype=torch.bool,
            device="cpu",
        )
        observations = {
            "pointgoal_with_gps_compass": torch.zeros((1,2)),
            "rgb": torch.zeros((1, width, height, 3))
        }

        # observations = batch_obs([observations], device="cpu")
        input_names = ["pointgoal", "rgb", "hidden_state", "prev_action", "not_done_mask"]
        output_names = ["features", "hidden_state"]
        torch.onnx.export(actor_critic.net, (observations, hidden_state, prev_action, not_done_masks), "out.onnx", verbose=True, input_names=input_names, output_names=output_names)