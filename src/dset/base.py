import abc
from typing import List

import numpy as np


class BaseDset(abc.ABC):

  def get_obs_id(self, obs_seq: List[str]) -> np.ndarray:
    # Type check.
    if not isinstance(obs_seq, list):
      raise TypeError('`obs_seq` must be an instance of `list`.')
    if not all(map(lambda obs: isinstance(obs, str), obs_seq)):
      raise TypeError('`obs_seq` must be a list of `str`.')

    return np.array(map(lambda obs: self.obs2id[obs], obs_seq))

  def get_obs(self, obs_id_seq: np.ndarray) -> List[str]:
    # Type check.
    if not isinstance(obs_id_seq, np.ndarray):
      raise TypeError('`obs_id_seq` must be an instance of `np.ndarray`.')

    return list(map(lambda obs_id: self.id2obs[obs_id], obs_id_seq))

  def get_state_id(self, state_seq: List[str]) -> np.ndarray:
    # Type check.
    if not isinstance(state_seq, list):
      raise TypeError('`state_seq` must be an instance of `list`.')
    if not all(map(lambda state: isinstance(state, str), state_seq)):
      raise TypeError('`state_seq` must be a list of `str`.')

    return np.array(map(lambda state: self.state2id[state], state_seq))

  def get_state(self, state_id_seq: np.ndarray) -> List[str]:
    # Type check.
    if not isinstance(state_id_seq, np.ndarray):
      raise TypeError('`state_id_seq` must be an instance of `np.ndarray`.')

    return list(map(lambda state_id: self.id2state[state_id], state_id_seq))
