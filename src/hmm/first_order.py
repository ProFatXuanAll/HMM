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

  def gen_seq(self, seq_len: int) -> np.ndarray:
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

  def forward_algo(self, obs_id_seq: np.ndarray) -> np.ndarray:
    # Type check.
    if not isinstance(obs_id_seq, np.ndarray):
      raise TypeError('`obs_id_seq` must be an instance of `np.ndarray`.')

    # Shape check.
    if len(obs_id_seq.shape) != 1:
      raise ValueError(f'The dimension of `obs_id_seq.shape` must be 1 instead of {len(obs_id_seq.shape)}')

    # Dimension check.
    if np.all((0 <= obs_id_seq) & (obs_id_seq <= self.n_obs - 1)):
      raise ValueError(f'Each entry in observation sequence must be within range [0, {self.n_obs - 1}].')

    seq_len = obs_id_seq.shape[0]
    forward_prob_mat = np.zeros((seq_len, self.n_state))

    # Calculate the probability of the initial state times the initial observation.
    forward_prob_mat[0] = self.init_m * self.emit_m[obs_id_seq[0]]

    # Iteratively calculate forward probabilities.
    for step in range(1, seq_len):
      forward_prob_vec = (forward_prob_mat[step - 1].reshape(-1, 1) * self.tran_m).sum(axis=0)
      forward_prob_mat[step] = (forward_prob_vec * self.emit_m[:, obs_id_seq[step]])

    return forward_prob_mat

  def obs_prob_by_forward(self, obs_id_seq: np.ndarray) -> float:
    forward_prob_mat = self.forward(obs_id_seq=obs_id_seq)
    return forward_prob_mat[-1].sum()

  def backward(self, obs_id_seq: np.ndarray) -> np.ndarray:
    total_time_step = obs_id_seq.shape[0]
    backward_prob_mat = np.zeros((total_time_step, self.n_state))

    # Initialize last time step T.
    backward_prob_mat[-1] = 1

    # Iteratively calculate backward probabilities \beta_t(i).
    for time_step in range(total_time_step - 2, -1, -1):
      backward_prob_mat[time_step] = (
        self.tran_m * self.emit_m[:, obs_id_seq[time_step + 1]].reshape(1, -1) *
        backward_prob_mat[time_step + 1].reshape(1, -1)
      ).sum(axis=1)

    return backward_prob_mat

  def obs_prob_by_backward(self, obs_id_seq: np.ndarray) -> float:
    backward_prob_mat = self.backward(obs_id_seq=obs_id_seq)
    return (self.init_m * self.emit_m[:, obs_id_seq[0]] * backward_prob_mat[0]).sum()

  def viterbi(self, obs_id_seq: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    total_time_step = obs_id_seq.shape[0]
    viterbi_prob_mat = np.zeros((total_time_step, self.n_state))
    prev_state_mat = np.zeros((total_time_step, self.n_state), dtype=np.int64)

    # Initialize time step 0.
    viterbi_prob_mat[0] = self.init_m * self.emit_m[:, obs_id_seq[0]]

    # Iteratively calculate Viterbi probabilities \delta_t(i) and previous best state
    # \psi_t(i).
    for time_step in range(1, total_time_step):
      viterbi_prob_vec = (viterbi_prob_mat[time_step - 1].reshape(-1, 1) * self.tran_m)
      prev_state_mat[time_step] = np.argmax(viterbi_prob_vec, axis=0)
      viterbi_prob_vec = np.max(viterbi_prob_vec, axis=0)
      viterbi_prob_mat[time_step] = (viterbi_prob_vec * self.emit_m[:, obs_id_seq[time_step]])

    # Back track all best states.
    all_best_states = [np.argmax(prev_state_mat[-1])]
    for time_step in range(total_time_step - 2, -1, -1):
      all_best_states.append(prev_state_mat[time_step, all_best_states[-1]])

    all_best_states.reverse()

    return viterbi_prob_mat, np.array(all_best_states)

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
