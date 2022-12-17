from typing import Tuple

import numpy as np

import hmm.algo.base
import hmm.algo.check

# Compatibility.
gen_sid_seq = hmm.algo.base.gen_sid_seq
gen_oid_seq = hmm.algo.base.gen_oid_seq
get_oid_seq_and_sid_seq_prob = hmm.algo.base.get_oid_seq_and_sid_seq_prob


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
  α[0] = init_m * emit_m[:, cur_oid]

  # Loop through time.
  for t in range(1, seq_len):
    cur_oid = oid_seq[t]

    # 1. (n_state) -> (1, n_state)
    # 2. (1, n_state) @ (n_state, n_state) -> (1, n_state)
    # 3. (1, n_state) -> (n_state)
    # 4. (n_state) * (n_state) -> (n_state)
    α[t] = (α[t - 1].reshape(1, n_state) @ tran_m).reshape(n_state) * emit_m[:, cur_oid]

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
  prob = α[-1].sum()
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

    # 1. (n_state) * (n_state) -> (n_state) -> (n_state, 1)
    # 2. (n_state, n_state) @ (n_state, 1) -> (n_state, 1) -> (n_state)
    β[t] = (tran_m @ (emit_m[:, next_oid] * β[t + 1]).reshape(n_state, 1)).reshape(n_state)

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
  prob = (init_m * emit_m[:, oid_seq[0]] * β[0]).sum()
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
  δ[0] = init_m * emit_m[:, cur_oid]

  # Loop through time.
  for t in range(1, seq_len):
    cur_oid = oid_seq[t]

    # 1. (n_state) -> (n_state, 1)
    # 2. (n_state, 1) * (n_state, n_state) -> (n_state, n_state)
    acc_prob = δ[t - 1].reshape(n_state, 1) * tran_m
    δ[t] = np.max(acc_prob, axis=0) * emit_m[:, cur_oid]
    ψ[t] = np.argmax(acc_prob, axis=0)

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
  γ = α * β

  # Loop through time.
  for t in range(seq_len - 1):
    next_oid = oid_seq[t + 1]
    for sid in range(n_state):
      ξ[t, sid] = α[t, sid] * tran_m[sid] * emit_m[:, next_oid] * β[t + 1]

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
  γ_sum = γ.sum(axis=0)
  new_tran_m = ξ.sum(axis=0) / (γ_sum - γ[seq_len - 1]).reshape(n_state, 1)

  # Reestimate emission probability.
  # 1. (n_obs) -> (1, n_obs)
  # 2. (t) -> (t, 1)
  # 3. (1, n_obs) == (t, 1) -> (t, n_obs)
  mask = np.arange(n_obs).reshape(1, n_obs) == oid_seq.reshape(seq_len, 1)
  # 1. (t, n_state) tranpose(1, 0) -> (n_state, t)
  # 2. (n_state, t) @ (t, n_obs) -> (n_state, n_obs)
  # 3. (n_state, n_obs) / (n_state, 1) -> (n_state, n_obs)
  new_emit_m = γ.transpose(1, 0) @ mask / γ_sum.reshape(n_state, 1)

  hmm.algo.check.check_hmm_param(emit_m=new_emit_m, init_m=new_init_m, tran_m=new_tran_m)
  return (new_emit_m, new_init_m, new_tran_m)
