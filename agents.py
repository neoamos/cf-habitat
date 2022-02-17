import numpy as np
import quaternion
import torch
import gym
from gym import spaces
from PIL import Image

from habitat_baselines.utils.common import (
    batch_obs,
)

from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.config.default import get_config
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
    apply_obs_transforms_obs_space,
    get_active_obs_transforms,
)

from habitat.tasks.utils import cartesian_to_polar
from habitat.utils.geometry_utils import (
    quaternion_rotate_vector,
)

from habitat_sim.utils.common import quat_from_two_vectors, quat_rotate_vector
from habitat_sim import geo

from habitat.core.spaces import ActionSpace, EmptySpace

class PointGoalAgent:
  def __init__(self, config_file, pretrained_weights, device="cpu"):
    self.device = device
    self.config = get_config(config_file)
    self.task_config = self.config.TASK_CONFIG
    self.possible_actions = self.task_config.TASK.POSSIBLE_ACTIONS
    self.config.freeze()

    self.observation_space = spaces.Dict({
      "pointgoal_with_gps_compass": spaces.Box(
        low=-3.4028235e+38, high=3.4028235e+38, shape=(2,)
        ),
      "rgb": spaces.Box(low=0, high=255, shape=(256, 256, 3), dtype=np.uint8)
    })
    self.action_space = ActionSpace(
      {a : EmptySpace() for a in self.possible_actions}
    )
    self.action_shape = (1,)
    self.action_type = torch.long
    self.obs_transforms = get_active_obs_transforms(self.config)

    self.observation_space = apply_obs_transforms_obs_space(
      self.observation_space, self.obs_transforms
    )

    policy = baseline_registry.get_policy(self.config.RL.POLICY.name)
    self.actor_critic = policy.from_config(
      self.config, self.observation_space, self.action_space
    )
    
    pretrained_state = torch.load(
      pretrained_weights, map_location="cpu"
    )
    self.actor_critic.eval()

    self.actor_critic.load_state_dict(
      {
          k[len("actor_critic.") :]: v
          for k, v in pretrained_state["state_dict"].items()
      }
    )

    self.reset_state()

  def reset_state(self):
    self.hidden_state = torch.zeros(
      1,
      self.actor_critic.net.num_recurrent_layers,
      self.config.RL.PPO.hidden_size,
      device=self.device,
    )
    self.prev_action = torch.zeros(
      1,
      *self.action_shape,
      device=self.device,
      dtype=self.action_type,
    )
    self.not_done_masks = torch.tensor(
        [False],
        dtype=torch.bool,
        device="cpu",
    )

  def act(self, image, position, yaw, target_position, point_goal=None):
    if point_goal is None:
      point_goal = compute_pointgoal_cf(position, yaw, target_position)
      point_goal = torch.from_numpy(np.expand_dims(point_goal, axis=0))
      image = torch.from_numpy(np.expand_dims(image, axis=0))

    observations = {
      "pointgoal_with_gps_compass": point_goal,
      "rgb": image
    }

    batch = batch_obs([observations], device=self.device)
    
    with torch.no_grad():
      (
        values,
        actions,
        actions_log_probs,
        hidden_state
      ) = self.actor_critic.act(
        batch,
        self.hidden_state,
        self.prev_action,
        self.not_done_masks,
        deterministic=False
      )

      self.hidden_state = hidden_state
      self.prev_action.copy_(actions)

    action = self.possible_actions[actions.item()]   

    self.not_done_masks = torch.tensor(
        [action!="STOP"],
        dtype=torch.bool,
        device="cpu",
    )
    return action

def compute_pointgoal_cf(
  source_position, yaw, goal_position
):
  rotation = quaternion.from_euler_angles(np.pi*((-yaw)/180), 0, 0)
  direction_vector = goal_position - source_position
  direction_vector_agent = quaternion_rotate_vector(
      rotation.inverse(), direction_vector
  )

  rho, phi = cartesian_to_polar(
      direction_vector_agent[0], direction_vector_agent[1]
  )
  return np.array([rho, -phi], dtype=np.float32)

def compute_pointgoal(
    source_position, source_rotation, goal_position, 
    format="POLAR", dimensionality=2
):
    direction_vector = goal_position - source_position
    direction_vector_agent = quaternion_rotate_vector(
        source_rotation.inverse(), direction_vector
    )

    if format == "POLAR":
        if dimensionality == 2:
            rho, phi = cartesian_to_polar(
                -direction_vector_agent[2], direction_vector_agent[0]
            )
            return np.array([rho, -phi], dtype=np.float32)
        else:
            _, phi = cartesian_to_polar(
                -direction_vector_agent[2], direction_vector_agent[0]
            )
            theta = np.arccos(
                direction_vector_agent[1]
                / np.linalg.norm(direction_vector_agent)
            )
            rho = np.linalg.norm(direction_vector_agent)

            return np.array([rho, -phi, theta], dtype=np.float32)
    else:
        if dimensionality == 2:
            return np.array(
                [-direction_vector_agent[2], direction_vector_agent[0]],
                dtype=np.float32,
            )
        else:
            return direction_vector_agent

def position_cf_to_hab(position):
  rotation_matrix = np.array([
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 0]
  ])
  quat = quaternion.from_rotation_matrix(rotation_matrix)
  return quaternion.rotate_vectors(quat, position)

def rotation_cf_to_hab(rotation):
  rotation_matrix = np.array([
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 0]
  ])
  quat = quaternion.from_rotation_matrix(rotation_matrix)
  return quat * rotation


if __name__ == "__main__":
  agent = PointGoalAgent(
    "configs/experiments/crazyflie_baseline_rgb.yaml",
    "ckpt.95.pth"
    )

  image = Image.open('out.jpg').resize((256,256))
  image = np.array(image)
  image = np.stack((image, image, image), axis=-1)
  image = image[:, :, 0:3]
  # image = np.zeros((256,256,3))
  position = np.array([0, 0, 0])
  yaw = 0
  target = np.array([1, 0, 0])

  for i in range(10):
    action = agent.act(image, position, yaw, target)
    print(action)

  yaw = 2.5
  position = np.array([-0.03, -0.05, 0.33])
  target = np.array([1, 0, 0])
  distance, rotation = compute_pointgoal_cf(position, yaw, target)
  print(distance, rotation*180/np.pi)
