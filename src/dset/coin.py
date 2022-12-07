import numpy as np

import src.dset.base
import src.util.func


class SimpleCoinTossTrials(src.dset.base.BaseDset):
  """Simple coin toss trials.

  There are 3 coins with probability of head equal to 0.1, 0.3, 0.5, respectively.
  All 3 coins have equal probability being chosen as the initial coin.
  When tossing the 0th coin, the next coin can only be the 1st or 2nd coin, each with equal probability.
  When tossing the 1th coin, the next coin can only be the 0th or 2nd coin, each with equal probability.
  When tossing the 2th coin, the next coin can only be the 0th or 1st coin, each with equal probability.
  """

  def __init__(self):
    self.obs2id = {
      'head': 0,
      'tail': 1,
    }
    self.id2obs = {
      0: 'head',
      1: 'tail',
    }
    self.state2id = {
      '0th-coin': 0,
      '1st-coin': 1,
      '2nd-coin': 2,
    }
    self.id2state = {
      0: '0th-coin',
      1: '1st-coin',
      2: '2nd-coin',
    }
    self.emit_m = np.array([
      [0.1, 0.9],
      [0.3, 0.7],
      [0.5, 0.5],
    ])
    self.init_m = np.array([1.0, 1.0, 1.0]) / 3
    self.tran_m = np.array([
      [0.0, 0.5, 0.5],
      [0.5, 0.0, 0.5],
      [0.5, 0.5, 0.0],
    ])


class RandomCoinTossTrials(src.dset.base.BaseDset):

  def __init__(self, *, n_coin: int, mean: float = 0.0, std: float = 1.0):
    # Type check.
    if not isinstance(n_coin, int):
      raise TypeError('`n_coin` must be an instance of `int`.')
    if not isinstance(mean, float):
      raise TypeError('`mean` must be an instance of `float`.')
    if not isinstance(std, float):
      raise TypeError('`std` must be an instance of `float`.')

    # Value check.
    if n_coin < 1:
      raise ValueError('`n_coin` must be a positive integer.')
    if std <= 0.0:
      raise ValueError('`std` must be a positive number.')

    self.obs2id = {
      'head': 0,
      'tail': 1,
    }
    self.id2obs = {
      0: 'head',
      1: 'tail',
    }
    self.state2id = {f'coin-{idx}': idx for idx in range(n_coin)}
    self.id2state = {idx: f'coin-{idx}' for idx in range(n_coin)}

    self.n_coin = n_coin
    self.mean = mean
    self.std = std

    # Generate random numbers.
    emit_m = np.random.normal(loc=mean, scale=std, size=(2, n_coin))
    init_m = np.random.normal(loc=mean, scale=std, size=(n_coin,))
    tran_m = np.random.normal(loc=mean, scale=std, size=(n_coin, n_coin))

    # Normalize to probability distribution.
    emit_m = src.util.func.softmax(arr=emit_m, axis=1)
    init_m = src.util.func.softmax(arr=init_m, axis=0)
    tran_m = src.util.func.softmax(arr=tran_m, axis=1)

    self.emit_m = emit_m
    self.init_m = init_m
    self.tran_m = tran_m
