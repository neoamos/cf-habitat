import numpy as np
import torch

from gym import spaces

from habitat_baseline.common.obs_transormers import ObservationTransformer


class RGBToGrayscale(ObservationTransformer):
  """
  A transformer that converts RGB images to grayscale
  """

  def __init__(
    self,
    size: int,
    channels_last: bool = True,
    trans_keys: Tuple[str, ...] = ("rgb", "depth", "semantic"),
):
    """Args:
    size: The size you want to resize the shortest edge to
    channels_last: indicates if channels is the last dimension
    """
    super(ResizeShortestEdge, self).__init__()
    self._size: int = size
    self.channels_last: bool = channels_last
    self.trans_keys: Tuple[str, ...] = trans_keys