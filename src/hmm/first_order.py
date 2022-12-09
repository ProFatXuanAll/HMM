from typing import Tuple

import numpy as np


class FirstOrderHMM:
  """First order hidden markov model.

  Parameters
  ==========
  emit_m: numpy.ndarray
    Emission probability matrix with shape (n_state, n_obs).
    The i-th row stands for the emission probability vector that corresponds to state i.
    The i-th row and the j-th column stands for the probability that observe j at state i.
    Each row must sum up to 1.
  init_m: numpy.ndarray
    Initial state probability matrix with shape (n_state).
    This matrix sum up to 1.
  tran_m: numpy.ndarray
    Transition probability matrix with shape (n_state, n_state).
    The i-th row stands for the transition probability vector that corresponds to state i.
    The i-th row and the j-th column stands for the probability that transit from state i to state j.
    Each row must sum up to 1.
  """

  def __init__(
    self,
    emit_m: np.ndarray,
    init_m: np.ndarray,
    tran_m: np.ndarray,
  ):
    # Type check.
    if not isinstance(emit_m, np.ndarray):
      raise TypeError('`emit_m` must be an instance of `np.ndarray`.')

    if not isinstance(init_m, np.ndarray):
      raise TypeError('`init_m` must be an instance of `np.ndarray`.')

    if not isinstance(tran_m, np.ndarray):
      raise TypeError('`tran_m` must be an instance of `np.ndarray`.')

    # Shape check.
    if len(emit_m.shape) != 2:
      raise ValueError(f'The dimension of `emit_m.shape` must be 2 instead of {len(emit_m.shape)}.')

    if len(init_m.shape) != 1:
      raise ValueError(f'The dimension of `init_m.shape` must be 1 instead of {len(init_m.shape)}.')

    if len(tran_m.shape) != 2:
      raise ValueError(f'The dimension of `tran_m.shape` must be 2 instead of {len(tran_m.shape)}.')

    n_state = init_m.shape[0]
    n_obs = emit_m.shape[1]

    # Dimension check.
    if emit_m.shape != (n_state, n_obs):
      raise ValueError(f'The dimension of `emit_m` must be {(n_state, n_obs)} instead of {emit_m.shape}.')
    if tran_m.shape != (n_state, n_state):
      raise ValueError(f'The dimension of `tran_m` must be {(n_state, n_state)} instead of {tran_m.shape}.')

    # Value check.
    if not np.all((0.0 <= emit_m) & (emit_m <= 1.0)):
      raise ValueError('Each entry in emission probability matrix must be within range [0, 1].')
    if not np.all((0.0 <= init_m) & (init_m <= 1.0)):
      raise ValueError('Each entry in initial state probability matrix must be within range [0, 1].')
    if not np.all((0.0 <= tran_m) & (tran_m <= 1.0)):
      raise ValueError('Each entry in transition probability matrix must be within range [0, 1].')

    if not np.all(np.isclose(emit_m.sum(axis=1), 1.0)):
      raise ValueError('Each row of emission probability matrix must sum up to 1.')
    if not np.all(np.isclose(init_m.sum(), 1.0)):
      raise ValueError('Initial state probability matrix must sum up to 1.')
    if not np.all(np.isclose(tran_m.sum(axis=1), 1.0)):
      raise ValueError('Each row of transition probability matrix must sum up to 1.')

    self.emit_m = emit_m
    self.init_m = init_m
    self.n_state = n_state
    self.n_obs = n_obs
    self.tran_m = tran_m

  def gen_seq(self, seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
    """Randomly generate sequence.

    Parameters
    ==========
    seq_len: int
      Length of randomly generated sequence.

    Returns
    =======
    tuple[np.ndarray, np.ndarray]
      First item in the returned tuple is the randomly generated sequence of states.
      Second item in the returned tuple is the randomly generated sequence of observations.
      The observation sequence is conditioned on the state sequence.
    """
    # Type check.
    if not isinstance(seq_len, int):
      raise TypeError('`seq_len` must be an instance of `int`.')

    # Value check.
    if seq_len < 1:
      raise ValueError('`seq_len` must be a positive integer.')

    obs_id_seq = np.empty((seq_len,), dtype=np.int64)
    state_id_seq = np.empty((seq_len,), dtype=np.int64)

    # Generate initial state.
    cur_state_id = np.random.choice(a=self.n_state, p=self.init_m)
    state_id_seq[0] = cur_state_id

    # Generate initial observation based on the initial state.
    obs_id_seq[0] = np.random.choice(a=self.n_obs, p=self.emit_m[cur_state_id])

    for step in range(1, seq_len):
      # Transit state.
      cur_state_id = np.random.choice(a=self.n_state, p=self.tran_m[cur_state_id])
      state_id_seq[step] = cur_state_id

      # Generate observation based on the current state.
      obs_id_seq[step] = np.random.choice(a=self.n_obs, p=self.emit_m[cur_state_id])

    return (state_id_seq, obs_id_seq)

  def check_obs_id_seq(self, obs_id_seq: np.ndarray) -> None:
    """Validate sequence of observation ids.

    Parameters
    ==========
    obs_id_seq: np.ndarray
      Sequence of observation ids to be validated.

    Raises
    ======
    TypeError
      When `obs_id_seq` is not an instance of `np.ndarray`.
    ValueError
      When the dimension of `obs_id_seq.shape` is not 1, or when some ids in `obs_id_seq` is out of range.
    """
    # Type check.
    if not isinstance(obs_id_seq, np.ndarray):
      raise TypeError('`obs_id_seq` must be an instance of `np.ndarray`.')

    # Dimension check.
    if len(obs_id_seq.shape) != 1:
      raise ValueError(f'The dimension of `obs_id_seq.shape` must be 1 instead of {len(obs_id_seq.shape)}.')

    # Value check.
    if not np.all((0 <= obs_id_seq) & (obs_id_seq <= self.n_obs - 1)):
      raise ValueError(f'`obs_id_seq` must sequence of integer within the range [0, {self.n_obs - 1}].')

  def check_state_id_seq(self, state_id_seq: np.ndarray) -> None:
    """Validate sequence of state ids.

    Parameters
    ==========
    state_id_seq: np.ndarray
      Sequence of state ids to be validated.

    Raises
    ======
    TypeError
      When `state_id_seq` is not an instance of `np.ndarray`.
    ValueError
      When the dimension of `state_id_seq.shape` is not 1, or when some ids in `state_id_seq` is out of range.
    """
    # Type check.
    if not isinstance(state_id_seq, np.ndarray):
      raise TypeError('`state_id_seq` must be an instance of `np.ndarray`.')

    # Dimension check.
    if len(state_id_seq.shape) != 1:
      raise ValueError(f'The dimension of `state_id_seq.shape` must be 1 instead of {len(state_id_seq.shape)}.')

    # Value check.
    if not np.all((0 <= state_id_seq) & (state_id_seq <= self.n_state - 1)):
      raise ValueError(f'`state_id_seq` must sequence of integer within the range [0, {self.n_state - 1}].')

  def get_obs_n_state_prob(self, obs_id_seq: np.ndarray, state_id_seq: np.ndarray) -> float:
    """Calculate the joint probability of the given observation and state sequences.

    Parameters
    ==========
    obs_id_seq: np.ndarray
      Sequence of observation ids to calculate the joint probability.
    state_id_seq: np.ndarray
      Sequence of state ids to calculate the joint probability.

    Returns
    =======
    float
      The joint probability of the given observation and state sequences.
    """
    self.check_obs_id_seq(obs_id_seq=obs_id_seq)
    self.check_state_id_seq(state_id_seq=state_id_seq)

    if not obs_id_seq.shape[0] != state_id_seq.shape[0]:
      raise ValueError('Length inconsistency.')

    seq_len = obs_id_seq.shape[0]

    cur_state_id = state_id_seq[0]
    cur_obs_id = obs_id_seq[0]
    prob = self.init_m[cur_state_id] * self.emit_m[cur_state_id, cur_obs_id]

    for cur_step in range(1, seq_len):
      prev_state_id = cur_state_id
      cur_state_id = state_id_seq[cur_step]
      cur_obs_id = obs_id_seq[cur_step]
      prob *= self.tran_m[prev_state_id, cur_state_id] * self.emit_m[cur_state_id, cur_obs_id]

    return prob

  def forward_algo(self, obs_id_seq: np.ndarray) -> Tuple[float, np.ndarray]:
    """Use forward algorithm to calculate observation probability.

    Parameters
    ==========
    obs_id_seq: np.ndarray
      Sequence of observation ids to calculate the probability.

    Returns
    =======
    tuple[float, np.ndarray]
      The first item in the returned tuple is the probability of the given observation sequence.
      The second item in the returned tuple is the forward probability sequence.
    """
    # Validate `obs_id_seq`.
    self.check_obs_id_seq(obs_id_seq=obs_id_seq)

    seq_len = obs_id_seq.shape[0]
    alpha_m = np.zeros((seq_len, self.n_state))

    cur_obs_id = obs_id_seq[0]
    alpha_m[0, :] = self.init_m * self.emit_m[:, cur_obs_id]

    for cur_step in range(1, seq_len):
      prev_step = cur_step - 1
      cur_obs_id = obs_id_seq[cur_step]

      # (1, n_state) @ (n_state, n_state) -> (1, n_state) -> (n_state)
      acc_prob = (alpha_m[prev_step, :].reshape(1, self.n_state) @ self.tran_m).reshape(self.n_state)
      alpha_m[cur_step, :] = acc_prob * self.emit_m[:, cur_obs_id]

    obs_prob = alpha_m[seq_len - 1].sum()
    return obs_prob, alpha_m

  def backward_algo(self, obs_id_seq: np.ndarray) -> Tuple[float, np.ndarray]:
    """Use backward algorithm to calculate observation probability.

    Parameters
    ==========
    obs_id_seq: np.ndarray
      Sequence of observation ids to calculate the probability.

    Returns
    =======
    tuple[float, np.ndarray]
      The first item in the returned tuple is the probability of the given observation sequence.
      The second item in the returned tuple is the backward probability sequence.
    """
    # Validate `obs_id_seq`.
    self.check_obs_id_seq(obs_id_seq=obs_id_seq)

    seq_len = obs_id_seq.shape[0]
    beta_m = np.ones((seq_len, self.n_state))

    for cur_step in range(seq_len - 2, -1, -1):
      next_step = cur_step + 1
      next_obs_id = obs_id_seq[next_step]

      acc_prob = self.emit_m[:, next_obs_id] * beta_m[next_step, :]
      # (n_state, n_state) @ (n_state, 1) -> (n_state, 1) -> (n_state)
      beta_m[cur_step, :] = (self.tran_m @ acc_prob.reshape(self.n_state, 1)).reshape(self.n_state)

    cur_obs_id = obs_id_seq[0]
    obs_prob = (beta_m[0, :] * self.emit_m[:, cur_obs_id] * self.init_m).sum()

    return obs_prob, beta_m

  def viterbi_algo(self, obs_id_seq: np.ndarray) -> Tuple[float, np.ndarray]:
    # Validate `obs_id_seq`.
    self.check_obs_id_seq(obs_id_seq=obs_id_seq)

    seq_len = obs_id_seq.shape[0]
    delta_m = np.zeros((seq_len, self.n_state))
    psi_m = np.zeros((seq_len, self.n_state), dtype=np.int64)

    cur_obs_id = obs_id_seq[0]
    delta_m[0, :] = self.init_m * self.emit_m[:, cur_obs_id]

    for cur_step in range(1, seq_len):
      prev_step = cur_step - 1
      cur_obs_id = obs_id_seq[cur_step]

      # (n_state, 1) * (n_state, n_state) -> (n_state, n_state)
      acc_prob = delta_m[prev_step, :].reshape(self.n_state, 1) * self.tran_m
      delta_m[cur_step, :] = np.max(acc_prob, axis=0) * self.emit_m[:, cur_obs_id]
      psi_m[cur_step, :] = np.argmax(acc_prob, axis=0)

    best_state_id_seq = np.zeros(seq_len, dtype=np.int64)
    cur_best_state_id = np.argmax(delta_m[seq_len - 1, :])
    best_state_id_seq[seq_len - 1] = cur_best_state_id
    best_obs_n_state_prob = delta_m[seq_len - 1, cur_best_state_id]

    for cur_step in range(seq_len - 2, -1, -1):
      next_step = cur_step + 1
      cur_best_state_id = psi_m[next_step, cur_best_state_id]
      best_state_id_seq[cur_step] = cur_best_state_id

    return best_obs_n_state_prob, best_state_id_seq

  def baum_welch(self, obs_id_seq: np.ndarray) -> None:
    total_time_step = obs_id_seq.shape[0]
    forward_prob_mat = self.forward(obs_id_seq=obs_id_seq)
    backward_prob_mat = self.backward(obs_id_seq=obs_id_seq)
    obs_prob = self.obs_prob_by_forward(obs_id_seq=obs_id_seq)

    # Iteratively calculate expected number of times transition from state i to state j at time
    # step t \xi_t(i, j). Also calculate expected number of times transistion from state i at
    # time step t \gamma_t(i).
    exp_from_i_to_j = np.zeros((total_time_step - 1, self.n_state, self.n_state))
    exp_from_i = np.zeros((total_time_step, self.n_state))

    for time_step in range(total_time_step - 1):
      exp_from_i_to_j[time_step] = (
        forward_prob_mat[time_step].reshape(-1, 1) * self.tran_m *
        self.emit_m[:, obs_id_seq[time_step + 1]].reshape(1, -1) * backward_prob_mat[time_step + 1].reshape(1, -1)
      ) / obs_prob
      exp_from_i[time_step] = exp_from_i_to_j[time_step].sum(axis=-1)

    exp_from_i[-1] = forward_prob_mat[-1] * backward_prob_mat[-1] / obs_prob

    # Update parameters.
    self.init_m = exp_from_i[0]
    self.tran_m = (exp_from_i_to_j.sum(axis=0) / exp_from_i[:-1].sum(axis=0).reshape(-1, 1))
    self.emit_m = (
      exp_from_i.T @ (obs_id_seq.reshape(-1, 1) == np.arange(self.n_obs).reshape(1, -1)) /
      exp_from_i.sum(axis=0).reshape(-1, 1)
    )
