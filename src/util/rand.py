import numpy as np
import random


def set_seed(seed: int) -> None:
  """Set random seed.

  Parameters
  ==========
  seed: int
    Random seed to be set.
  """
  random.seed(seed)
  np.random.seed(seed)
