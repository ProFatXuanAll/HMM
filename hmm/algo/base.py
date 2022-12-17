from typing import Tuple

import numpy as np

import hmm.algo.check


def gen_sid_seq(
  init_m: np.ndarray,
  seq_len: int,
  tran_m: np.ndarray,
) -> np.ndarray:
  """Randomly generate sequence of state ids.

  Parameters
  ==========
  init_m: numpy.ndarray
    Initial state probability matrix with shape (n_state).
    This matrix sum up to 1.
  seq_len: int
    Length of the randomly generated sequence of state ids.
  tran_m: numpy.ndarray
    Transition probability matrix with shape (n_state, n_state).
    The i-th row stands for the transition probability vector that corresponds to state i.
    The i-th row and the j-th column stands for the probability that transit from state i to state j.
    Each row must sum up to 1.

  Returns
  =======
  np.ndarray
    The randomly generated sequence of state ids.
  """
  hmm.algo.check.check_init_m(init_m=init_m)
  hmm.algo.check.check_seq_len(seq_len=seq_len)
  hmm.algo.check.check_trans_m(tran_m=tran_m)

  n_state = init_m.shape[0]
  sid_seq = np.zeros((seq_len,), dtype=np.int64)

  # Generate initial state.
  cur_sid = np.random.choice(a=n_state, p=init_m)
  sid_seq[0] = cur_sid

  for t in range(1, seq_len):
    # Transit state.
    cur_sid = np.random.choice(a=n_state, p=tran_m[cur_sid])
    sid_seq[t] = cur_sid

  return sid_seq


def gen_oid_seq(
  emit_m: np.ndarray,
  sid_seq: np.ndarray,
) -> np.ndarray:
  """Randomly generate sequence of observation ids.

  Parameters
  ==========
  emit_m: numpy.ndarray
    Emission probability matrix with shape (n_state, n_obs).
    The i-th row stands for the emission probability vector that corresponds to state i.
    The i-th row and the j-th column stands for the probability that observe j at state i.
    Each row must sum up to 1.
  sid_seq: numpy.ndarray
    Sequence of state ids to condition on.

  Returns
  =======
  np.ndarray
    The randomly generated sequence of observation ids.
  """
  hmm.algo.check.check_emit_m(emit_m=emit_m)
  n_state, n_obs = emit_m.shape

  hmm.algo.check.check_id_seq(id_seq=sid_seq, id_seq_name='sid_seq', max_id=n_state - 1, min_id=0)
  seq_len = sid_seq.shape[0]

  oid_seq = np.zeros((seq_len,), dtype=np.int64)

  # Generate initial observation based on the initial state.
  cur_sid = sid_seq[0]
  oid_seq[0] = np.random.choice(a=n_obs, p=emit_m[cur_sid])

  for t in range(1, seq_len):
    # Transit state.
    cur_sid = sid_seq[t]

    # Generate observation based on the current state.
    oid_seq[t] = np.random.choice(a=n_obs, p=emit_m[cur_sid])

  return oid_seq


def get_oid_seq_and_sid_seq_prob(
  emit_m: np.ndarray,
  init_m: np.ndarray,
  oid_seq: np.ndarray,
  sid_seq: np.ndarray,
  tran_m: np.ndarray,
) -> float:
  """Calculate the joint probability of the given observation and state sequences.

  Parameters
  ==========
  init_m: numpy.ndarray
    Initial state probability matrix with shape (n_state).
    This matrix sum up to 1.
  emit_m: numpy.ndarray
    Emission probability matrix with shape (n_state, n_obs).
    The i-th row stands for the emission probability vector that corresponds to state i.
    The i-th row and the j-th column stands for the probability that observe j at state i.
    Each row must sum up to 1.
  oid_seq: numpy.ndarray
    Sequence of observation ids to calculate the joint probability.
  sid_seq: np.ndarray
    Sequence of state ids to calculate the joint probability.
  tran_m: numpy.ndarray
    Transition probability matrix with shape (n_state, n_state).
    The i-th row stands for the transition probability vector that corresponds to state i.
    The i-th row and the j-th column stands for the probability that transit from state i to state j.
    Each row must sum up to 1.

  Returns
  =======
  float
    The joint probability of the given observation and state sequences.
  """
  hmm.algo.check.check_hmm_param(emit_m=emit_m, init_m=init_m, tran_m=tran_m)

  n_state, n_obs = emit_m.shape
  hmm.algo.check.check_id_seq(id_seq=oid_seq, id_seq_name='oid_seq', max_id=n_obs - 1, min_id=0)
  hmm.algo.check.check_id_seq(id_seq=sid_seq, id_seq_name='sid_seq', max_id=n_state - 1, min_id=0)

  if oid_seq.shape[0] != sid_seq.shape[0]:
    raise ValueError('Length inconsistency.')

  seq_len = oid_seq.shape[0]

  cur_sid = sid_seq[0]
  cur_oid = oid_seq[0]
  prob = init_m[cur_sid] * emit_m[cur_sid, cur_oid]

  for t in range(1, seq_len):
    prev_sid = cur_sid
    cur_sid = sid_seq[t]
    cur_oid = oid_seq[t]
    prob *= tran_m[prev_sid, cur_sid] * emit_m[cur_sid, cur_oid]

  return prob


def forward_algo(
  emit_m: np.ndarray,
  init_m: np.ndarray,
  oid_seq: np.ndarray,
  tran_m: np.ndarray,
) -> np.ndarray:
  """Use the forward algorithm to calculate the joint probability of observations o_0, ..., o_t and s_t.

  Parameters
  ==========
  init_m: numpy.ndarray
    Initial state probability matrix with shape (n_state).
    This matrix sum up to 1.
  emit_m: numpy.ndarray
    Emission probability matrix with shape (n_state, n_obs).
    The i-th row stands for the emission probability vector that corresponds to state i.
    The i-th row and the j-th column stands for the probability that observe j at state i.
    Each row must sum up to 1.
  oid_seq: numpy.ndarray
    Sequence of observation ids to calculate the joint probability.
  tran_m: numpy.ndarray
    Transition probability matrix with shape (n_state, n_state).
    The i-th row stands for the transition probability vector that corresponds to state i.
    The i-th row and the j-th column stands for the probability that transit from state i to state j.
    Each row must sum up to 1.

  Returns
  =======
  numpy.ndarray
    Sequence of the forward probability.
  """
  hmm.algo.check.check_hmm_param(emit_m=emit_m, init_m=init_m, tran_m=tran_m)

  n_state, n_obs = emit_m.shape
  hmm.algo.check.check_id_seq(id_seq=oid_seq, id_seq_name='oid_seq', max_id=n_obs - 1, min_id=0)

  seq_len = oid_seq.shape[0]
  α = np.zeros((seq_len, n_state))

  # Initialize α.
  cur_oid = oid_seq[0]
  for sid in range(n_state):
    α[0, sid] = init_m[sid] * emit_m[sid, cur_oid]

  # Loop through time.
  for t in range(1, seq_len):
    cur_oid = oid_seq[t]

    for dst_sid in range(n_state):
      # Multiplication with emit_m can be factored out.
      tmp = 0
      for src_sid in range(n_state):
        tmp += α[t - 1, src_sid] * tran_m[src_sid, dst_sid]
      α[t, dst_sid] = tmp * emit_m[dst_sid, cur_oid]

  return α


def get_oid_seq_prob_by_forward_algo(
  emit_m: np.ndarray,
  init_m: np.ndarray,
  oid_seq: np.ndarray,
  tran_m: np.ndarray,
) -> float:
  """Use the forward algorithm to calculate the probability of the sequence of observation ids.

  Parameters
  ==========
  init_m: numpy.ndarray
    Initial state probability matrix with shape (n_state).
    This matrix sum up to 1.
  emit_m: numpy.ndarray
    Emission probability matrix with shape (n_state, n_obs).
    The i-th row stands for the emission probability vector that corresponds to state i.
    The i-th row and the j-th column stands for the probability that observe j at state i.
    Each row must sum up to 1.
  oid_seq: numpy.ndarray
    Sequence of observation ids to calculate the probability.
  tran_m: numpy.ndarray
    Transition probability matrix with shape (n_state, n_state).
    The i-th row stands for the transition probability vector that corresponds to state i.
    The i-th row and the j-th column stands for the probability that transit from state i to state j.
    Each row must sum up to 1.

  Returns
  =======
  float
    The probability of the sequence of observation ids.
  """
  α = forward_algo(emit_m=emit_m, init_m=init_m, oid_seq=oid_seq, tran_m=tran_m)

  n_state = init_m.shape[0]
  seq_len = oid_seq.shape[0]

  prob = 0.0
  for sid in range(n_state):
    prob += α[seq_len - 1, sid]

  return prob


def backward_algo(
  emit_m: np.ndarray,
  init_m: np.ndarray,
  oid_seq: np.ndarray,
  tran_m: np.ndarray,
) -> np.ndarray:
  """Use the backward algorithm to calculate the probability of observations o_0, ..., o_t conditioned on s_t.

  Parameters
  ==========
  init_m: numpy.ndarray
    Initial state probability matrix with shape (n_state).
    This matrix sum up to 1.
  emit_m: numpy.ndarray
    Emission probability matrix with shape (n_state, n_obs).
    The i-th row stands for the emission probability vector that corresponds to state i.
    The i-th row and the j-th column stands for the probability that observe j at state i.
    Each row must sum up to 1.
  oid_seq: numpy.ndarray
    Sequence of observation ids to calculate the conditional probability.
  tran_m: numpy.ndarray
    Transition probability matrix with shape (n_state, n_state).
    The i-th row stands for the transition probability vector that corresponds to state i.
    The i-th row and the j-th column stands for the probability that transit from state i to state j.
    Each row must sum up to 1.

  Returns
  =======
  numpy.ndarray
    Sequence of the backward probability.
  """
  hmm.algo.check.check_hmm_param(emit_m=emit_m, init_m=init_m, tran_m=tran_m)

  n_state, n_obs = emit_m.shape
  hmm.algo.check.check_id_seq(id_seq=oid_seq, id_seq_name='oid_seq', max_id=n_obs - 1, min_id=0)

  seq_len = oid_seq.shape[0]
  β = np.zeros((seq_len, n_state))

  # Initialize β.
  for sid in range(n_state):
    β[seq_len - 1, sid] = 1.0

  # Loop through time.
  for t in range(seq_len - 2, -1, -1):
    next_oid = oid_seq[t + 1]

    # emit_m * β can be reused.
    tmp = np.zeros(n_state)
    for src_sid in range(n_state):
      tmp[src_sid] = emit_m[src_sid, next_oid] * β[t + 1, src_sid]

    for dst_sid in range(n_state):
      for src_sid in range(n_state):
        β[t, dst_sid] += tran_m[dst_sid, src_sid] * tmp[src_sid]

  return β


def get_oid_seq_prob_by_backward_algo(
  emit_m: np.ndarray,
  init_m: np.ndarray,
  oid_seq: np.ndarray,
  tran_m: np.ndarray,
) -> float:
  """Use the forward algorithm to calculate the probability of the sequence of observation ids.

  Parameters
  ==========
  init_m: numpy.ndarray
    Initial state probability matrix with shape (n_state).
    This matrix sum up to 1.
  emit_m: numpy.ndarray
    Emission probability matrix with shape (n_state, n_obs).
    The i-th row stands for the emission probability vector that corresponds to state i.
    The i-th row and the j-th column stands for the probability that observe j at state i.
    Each row must sum up to 1.
  oid_seq: numpy.ndarray
    Sequence of observation ids to calculate the probability.
  tran_m: numpy.ndarray
    Transition probability matrix with shape (n_state, n_state).
    The i-th row stands for the transition probability vector that corresponds to state i.
    The i-th row and the j-th column stands for the probability that transit from state i to state j.
    Each row must sum up to 1.

  Returns
  =======
  float
    The probability of the sequence of observation ids.
  """
  β = backward_algo(emit_m=emit_m, init_m=init_m, oid_seq=oid_seq, tran_m=tran_m)

  n_state = init_m.shape[0]
  cur_oid = oid_seq[0]

  prob = 0.0
  for sid in range(n_state):
    prob += (init_m[sid] * emit_m[sid, cur_oid] * β[0, sid])

  return prob


def viterbi_algo(
  emit_m: np.ndarray,
  init_m: np.ndarray,
  oid_seq: np.ndarray,
  tran_m: np.ndarray,
) -> np.ndarray:
  """Use the Viterbi algorithm to decode the most possible state sequence given the observation sequence.

  Parameters
  ==========
  init_m: numpy.ndarray
    Initial state probability matrix with shape (n_state).
    This matrix sum up to 1.
  emit_m: numpy.ndarray
    Emission probability matrix with shape (n_state, n_obs).
    The i-th row stands for the emission probability vector that corresponds to state i.
    The i-th row and the j-th column stands for the probability that observe j at state i.
    Each row must sum up to 1.
  oid_seq: numpy.ndarray
    Sequence of observation ids to calculate the joint probability.
  tran_m: numpy.ndarray
    Transition probability matrix with shape (n_state, n_state).
    The i-th row stands for the transition probability vector that corresponds to state i.
    The i-th row and the j-th column stands for the probability that transit from state i to state j.
    Each row must sum up to 1.

  Returns
  =======
  numpy.ndarray
    The sequence of state ids with the highest probability conditioned on the sequence of observation ids.
  """
  hmm.algo.check.check_hmm_param(emit_m=emit_m, init_m=init_m, tran_m=tran_m)

  n_state, n_obs = emit_m.shape
  hmm.algo.check.check_id_seq(id_seq=oid_seq, id_seq_name='oid_seq', max_id=n_obs - 1, min_id=0)

  seq_len = oid_seq.shape[0]
  δ = np.zeros((seq_len, n_state))
  ψ = np.zeros((seq_len, n_state), dtype=np.int64)

  # Initialize δ and ψ.
  cur_oid = oid_seq[0]
  for sid in range(n_state):
    δ[0, sid] = init_m[sid] * emit_m[sid, cur_oid]

  # Loop through time.
  for t in range(1, seq_len):
    cur_oid = oid_seq[t]

    for dst_sid in range(n_state):
      # Suppose the maximum probability is at state 0.
      δ[t, dst_sid] = δ[t - 1, 0] * tran_m[0, dst_sid]
      ψ[t, dst_sid] = 0
      # Loop to find the maximum state.
      for src_sid in range(1, n_state):
        tmp = δ[t - 1, src_sid] * tran_m[src_sid, dst_sid]
        if δ[t, dst_sid] < tmp:
          δ[t, dst_sid] = tmp
          ψ[t, dst_sid] = src_sid
      # Multiplication with emit_m can be factored out.
      δ[t, dst_sid] *= emit_m[dst_sid, cur_oid]

  # Reverse decode.
  best_sid_seq = np.zeros(seq_len, dtype=np.int64)
  cur_best_sid = np.argmax(δ[seq_len - 1])
  best_sid_seq[seq_len - 1] = cur_best_sid

  for t in range(seq_len - 2, -1, -1):
    cur_best_sid = ψ[t + 1, cur_best_sid]
    best_sid_seq[t] = cur_best_sid

  return best_sid_seq


def compute_ξ_and_γ(
  emit_m: np.ndarray,
  init_m: np.ndarray,
  oid_seq: np.ndarray,
  tran_m: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
  """Calculate the probability at specific state(s) conditioned on the observation sequence.

  ξ[t, i, j] is the probability of S_t = i and S_{t+1} = j conditioned on the observation sequence.
  γ[t, i]    is the probability of S_t = i conditioned on the observation sequence.

  Parameters
  ==========
  init_m: numpy.ndarray
    Initial state probability matrix with shape (n_state).
    This matrix sum up to 1.
  emit_m: numpy.ndarray
    Emission probability matrix with shape (n_state, n_obs).
    The i-th row stands for the emission probability vector that corresponds to state i.
    The i-th row and the j-th column stands for the probability that observe j at state i.
    Each row must sum up to 1.
  oid_seq: numpy.ndarray
    Sequence of observation ids to be used as the condition.
  tran_m: numpy.ndarray
    Transition probability matrix with shape (n_state, n_state).
    The i-th row stands for the transition probability vector that corresponds to state i.
    The i-th row and the j-th column stands for the probability that transit from state i to state j.
    Each row must sum up to 1.

  Returns
  =======
  Tuple[numpy.ndarray, numpy.ndarray]
    The first item in the returned tuple is ξ with shape (seq_len - 1, n_state, n_state).
    The second item in the returned tuple is γ with shape (seq_len, n_state).
  """
  α = forward_algo(emit_m=emit_m, init_m=init_m, oid_seq=oid_seq, tran_m=tran_m)
  β = backward_algo(emit_m=emit_m, init_m=init_m, oid_seq=oid_seq, tran_m=tran_m)
  prob = α[-1].sum()

  n_state = tran_m.shape[0]
  seq_len = oid_seq.shape[0]
  ξ = np.zeros((seq_len - 1, n_state, n_state))
  γ = np.zeros((seq_len, n_state))

  # First compute ξ.
  for t in range(seq_len - 1):
    next_oid = oid_seq[t + 1]
    for sid_i in range(n_state):
      γ[t, sid_i] = α[t, sid_i] * β[t, sid_i]
      for sid_j in range(n_state):
        ξ[t, sid_i, sid_j] = α[t, sid_i] * tran_m[sid_i, sid_j] * emit_m[sid_j, next_oid] * β[t + 1, sid_j]

  # Next compute γ.
  for t in range(seq_len):
    for sid_i in range(n_state):
      γ[t, sid_i] = α[t, sid_i] * β[t, sid_i]

  # Normalize by the probability observation sequence.
  ξ /= prob
  γ /= prob
  return ξ, γ


def baum_welch_algo(
  emit_m: np.ndarray,
  init_m: np.ndarray,
  oid_seq: np.ndarray,
  tran_m: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Reestimate a hidden markov model's parameters with Baum-Welch algorithm.

  Parameters
  ==========
  init_m: numpy.ndarray
    Initial state probability matrix with shape (n_state).
    This matrix sum up to 1.
  emit_m: numpy.ndarray
    Emission probability matrix with shape (n_state, n_obs).
    The i-th row stands for the emission probability vector that corresponds to state i.
    The i-th row and the j-th column stands for the probability that observe j at state i.
    Each row must sum up to 1.
  oid_seq: numpy.ndarray
    Sequence of observation ids to be used as the condition.
  tran_m: numpy.ndarray
    Transition probability matrix with shape (n_state, n_state).
    The i-th row stands for the transition probability vector that corresponds to state i.
    The i-th row and the j-th column stands for the probability that transit from state i to state j.
    Each row must sum up to 1.

  Returns
  =======
  numpy.ndarray
    Sequence of the two states probability.
  """
  ξ, γ = compute_ξ_and_γ(emit_m=emit_m, init_m=init_m, oid_seq=oid_seq, tran_m=tran_m)

  n_state, n_obs = emit_m.shape
  seq_len = oid_seq.shape[0]

  # Reestimate initial state probability.
  new_init_m = γ[0]

  # Reestimate transition probability.
  new_tran_m = np.zeros((n_state, n_state))
  nu_tran_m = np.zeros((n_state, n_state))
  de_tran_m = np.zeros((n_state,))

  for t in range(seq_len - 1):
    for sid_i in range(n_state):
      de_tran_m[sid_i] += γ[t, sid_i]
      for sid_j in range(n_state):
        nu_tran_m[sid_i, sid_j] += ξ[t, sid_i, sid_j]

  for sid_i in range(n_state):
    for sid_j in range(n_state):
      new_tran_m[sid_i, sid_j] = nu_tran_m[sid_i, sid_j] / de_tran_m[sid_i]

  # Reestimate emission probability.
  new_emit_m = np.zeros((n_state, n_obs))
  nu_emit_m = np.zeros((n_state, n_obs))
  de_emit_m = de_tran_m + γ[seq_len - 1]

  for t in range(seq_len):
    for sid in range(n_state):
      for oid in range(n_obs):
        if oid == oid_seq[t]:
          nu_emit_m[sid, oid] += γ[t, sid]

  for sid in range(n_state):
    for oid in range(n_obs):
      new_emit_m[sid, oid] = nu_emit_m[sid, oid] / de_emit_m[sid]

  hmm.algo.check.check_hmm_param(emit_m=new_emit_m, init_m=new_init_m, tran_m=new_tran_m)
  return (new_emit_m, new_init_m, new_tran_m)
