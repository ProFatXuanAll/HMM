import abc
from typing import Dict, List

import numpy as np


class BaseModel(abc.ABC):
  id2o: Dict[int, str]
  id2s: Dict[int, str]
  o2id: Dict[str, int]
  s2id: Dict[str, int]

  def get_oid_seq(self, o_seq: List[str]) -> np.ndarray:
    # Type check.
    if not isinstance(o_seq, list):
      raise TypeError('`o_seq` must be an instance of `list`.')
    if not all(map(lambda o: isinstance(o, str), o_seq)):
      raise TypeError('`o_seq` must be a list of `str`.')

    return np.array(map(lambda o: self.o2id[o], o_seq))

  def get_o_seq(self, oid_seq: np.ndarray) -> List[str]:
    # Type check.
    if not isinstance(oid_seq, np.ndarray):
      raise TypeError('`oid_seq` must be an instance of `numpy.ndarray`.')

    return list(map(lambda oid: self.id2o[oid], oid_seq))

  def get_sid_seq(self, s_seq: List[str]) -> np.ndarray:
    # Type check.
    if not isinstance(s_seq, list):
      raise TypeError('`s_seq` must be an instance of `list`.')
    if not all(map(lambda s: isinstance(s, str), s_seq)):
      raise TypeError('`s_seq` must be a list of `str`.')

    return np.array(map(lambda s: self.s2id[s], s_seq))

  def get_s_seq(self, sid_seq: np.ndarray) -> List[str]:
    # Type check.
    if not isinstance(sid_seq, np.ndarray):
      raise TypeError('`sid_seq` must be an instance of `numpy.ndarray`.')

    return list(map(lambda sid: self.id2s[sid], sid_seq))
