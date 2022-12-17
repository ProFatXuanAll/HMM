from typing import Tuple, Union

import numpy as np


def check_seq_len(seq_len: int) -> None:
  """Check sequence length.

  Parameters
  ==========
  seq_len: int
    Sequence length to be checked.

  Raises
  ======
  ValueError
    When `seq_len` is not a positive integer.
  """
  if seq_len < 1:
    raise ValueError('`seq_len` must be a positive integer.')


def check_is_vector(
  obj: np.ndarray,
  obj_name: str,
) -> None:
  """Check if the given numpy array is 1-dimensional.

  Parameters
  ==========
  obj: numpy.ndarray
    The numpy array to be checked.
  obj_name: str
    Name of the numpy array to be checked.

  Raises
  ======
  ValueError
    If `obj` is not a 1-dimensional numpy array.
  """
  if len(obj.shape) != 1:
    raise ValueError(f'`{obj_name}` is not a vector (1D array).')


def check_is_matrix(
  obj: np.ndarray,
  obj_name: str,
) -> None:
  """Check if the given numpy array is 2-dimensional.

  Parameters
  ==========
  obj: numpy.ndarray
    The numpy array to be checked.
  obj_name: str
    Name of the numpy array to be checked.

  Raises
  ======
  ValueError
    If `obj` is not a 2-dimensional numpy array.
  """
  if len(obj.shape) != 2:
    raise ValueError(f'`{obj_name}` is not a matrix (2D array).')


def check_is_probability(obj: np.ndarray, obj_name: str) -> None:
  """Check if each entry in the given numpy array is within the range of probability.

  Parameters
  ==========
  obj: numpy.ndarray
    The numpy array to be checked.
  obj_name: str
    Name of the numpy array to be checked.

  Raises
  ======
  ValueError
    If some entries in `obj` is not within the range of probability.
  """
  if not np.all(((0.0 <= obj) & (obj <= 1.0)) | np.isclose(obj, 0.0) | np.isclose(obj, 1.0)):
    raise ValueError(f'Each entry in {obj_name} must be within range [0, 1].')


def check_is_sum_to_1(axis: int, obj: np.ndarray, obj_name: str) -> None:
  """Check if the sum of the given numpy array along the given axis is equal to 1.

  Parameters
  ==========
  axis: int
    Axis to sum along.
  obj: numpy.ndarray
    The numpy array to be checked.
  obj_name: str
    Name of the numpy array to be checked.

  Raises
  ======
  ValueError
    If the sum along the axis is not equal to 1.
  """
  if not np.all(np.isclose(obj.sum(axis=axis), 1.0)):
    raise ValueError(f'{obj_name} must sum up to 1 alone axis {axis}.')


def check_shape(
  obj: np.ndarray,
  obj_name: str,
  shape: Union[Tuple[int], Tuple[int, int]],
) -> None:
  """Check if the shape of the given numpy array is as expected.

  Parameters
  ==========
  obj: numpy.ndarray
    The numpy array to be checked.
  obj_name: str
    Name of the numpy array to be checked.
  shape: typing.Union[tuple[int], tuple[int, int]]
    Expected shape of the given numpy array.

  Raises
  ======
  ValueError
    If `obj.shape` is not equal to `shape`.
  """
  if obj.shape != shape:
    raise ValueError(f'The shape of `{obj_name}` must be {shape} instead of {obj.shape}.')


def check_emit_m(emit_m: np.ndarray) -> None:
  """Check emission probability matrix.

  Parameters
  ==========
  emit_m: numpy.ndarray
    Emission probability matrix with shape (n_state, n_obs).
    The i-th row stands for the emission probability vector that corresponds to state i.
    The i-th row and the j-th column stands for the probability that observe j at state i.
    Each row must sum up to 1.

  Raises
  ======
  ValueError
    When `emit_m` is a invalid emission probability matrix.
  """
  obj_name = 'emission probability matrix'

  check_is_matrix(obj=emit_m, obj_name=obj_name)
  check_is_probability(obj=emit_m, obj_name=obj_name)
  check_is_sum_to_1(axis=1, obj=emit_m, obj_name=obj_name)


def check_init_m(init_m: np.ndarray) -> None:
  """Check initial state probability matrix.

  Parameters
  ==========
  init_m: numpy.ndarray
    Initial state probability matrix with shape (n_state).
    This matrix sum up to 1.

  Raises
  ======
  ValueError
    When `init_m` is a invalid initial state probability matrix.
  """
  obj_name = 'initial state probability matrix'

  check_is_vector(obj=init_m, obj_name=obj_name)
  check_is_probability(obj=init_m, obj_name=obj_name)
  check_is_sum_to_1(axis=0, obj=init_m, obj_name=obj_name)


def check_trans_m(tran_m: np.ndarray) -> None:
  """Check transition probability matrix.

  Parameters
  ==========
  tran_m: numpy.ndarray
    Transition probability matrix with shape (n_state, n_state).
    The i-th row stands for the transition probability vector that corresponds to state i.
    The i-th row and the j-th column stands for the probability that transit from state i to state j.
    Each row must sum up to 1.

  Raises
  ======
  ValueError
    When `tran_m` is a invalid transition probability matrix.
  """
  obj_name = 'transition probability matrix'

  check_is_matrix(obj=tran_m, obj_name=obj_name)
  check_is_probability(obj=tran_m, obj_name=obj_name)
  check_is_sum_to_1(axis=1, obj=tran_m, obj_name=obj_name)

  # Must be a square matrix.
  n_state = tran_m.shape[0]
  check_shape(obj=tran_m, obj_name=obj_name, shape=(n_state, n_state))


def check_hmm_param(
  emit_m: np.ndarray,
  init_m: np.ndarray,
  tran_m: np.ndarray,
) -> None:
  """Check parameters for a hidden markov model.

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

  Raises
  ======
  ValueError
    When one of `emit_m`, `init_m` or `tran_m` is invalid, or shape between the three are inconsistent.
  """
  check_emit_m(emit_m=emit_m)
  check_init_m(init_m=init_m)
  check_trans_m(tran_m=tran_m)

  n_state = init_m.shape[0]
  n_obs = emit_m.shape[1]

  # Ensure shape consistency.
  check_shape(obj=emit_m, obj_name='emission probability matrix', shape=(n_state, n_obs))
  check_shape(obj=tran_m, obj_name='transition probability matrix', shape=(n_state, n_state))


def check_id_seq(
  id_seq: np.ndarray,
  id_seq_name: str,
  max_id: int,
  min_id: int,
) -> None:
  """Check if the given sequence of ids is within the range [min_id, max_id].

  Parameters
  ==========
  id_seq: numpy.ndarray
    The sequence of id to be checked.
  id_seq_name: str
    Name of the sequence of id to be checked.
  max_id: int
    Largest id allowed in the given sequence (inclusive).
  min_id: int
    Smallest id allowed in the given sequence (inclusive).

  Raises
  ======
  ValueError
    When some id in the `id_seq` is not within the valid range.
  """
  check_is_vector(obj=id_seq, obj_name=id_seq_name)

  if not np.all((min_id <= id_seq) & (id_seq <= max_id)):
    raise ValueError(f'`{id_seq_name}` must a sequence of ids within the range [{min_id}, {max_id}].')
