import numpy as np
import torch
from torchvision import transforms

from gym import spaces
from typing import Dict, Iterable, List, Optional, Tuple, Union

from habitat.config import Config
from habitat_baselines.common.obs_transformers import ObservationTransformer
from habitat_baselines.common.baseline_registry import baseline_registry


@baseline_registry.register_obs_transformer()
class CrazyflieCamera(ObservationTransformer):
  """
  A transformer that converts RGB images to grayscale
  """

  def __init__(self,
    brightness = [0.8, 1.2],
    contrast = [0.7, 2.0]
  ):
    super(CrazyflieCamera, self).__init__()
    self._transformer = torch.nn.Sequential(
      transforms.ColorJitter(
        brightness = brightness,
        contrast = contrast
      ),
      transforms.Grayscale(3)
    )

  def transform_observation_space(
    self,
    observation_space: spaces.Dict
  ):
    return observation_space


  @torch.no_grad()
  def forward(
    self,
    observations: Dict[str, torch.Tensor]
  ):
    imgs = observations['rgb'].permute((0,3,1,2))
    imgs = self._transformer(imgs)
    observations['rgb'] = imgs.permute((0, 2, 3, 1))
    return observations

  @classmethod
  def from_config(cls, config: Config):
    config = config.RL.POLICY.OBS_TRANSFORMS.CrazyflieCamera
    return cls(
      brightness = config.BRIGHTNESS,
      contrast = config.CONTRAST
    )
