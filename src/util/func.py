import numpy as np


def softmax(*, arr: np.ndarray, axis: int = 0) -> np.ndarray:
  """Softmax operation implemented in `np.ndarray`.

  Parameters
  ==========
  arr: np.ndarray
    Perform softmax on the given array.
  axis: int
    Perform softmax on the specified axis.
  """
  # Type check.
  if not isinstance(arr, np.ndarray):
    raise ValueError('`arr` must be an instance of `np.ndarray`.')
  if not isinstance(axis, int):
    raise ValueError('`axis` must be an instance of `int`.')

  # Value check.
  if axis < 0:
    raise ValueError('`axis` must be a non-negative integer.')
  if len(arr.shape) < axis:
    raise ValueError('`axis` can be `len(arr.shape) - 1` at most.')

  # Convert numbers to non-negative values through exponential.
  numerator = np.exp(arr)

  # Get broadcast shape.
  # For example:
  # shape = (2, 3, 4), axis = 0, broadcast shape = (1, 3, 4).
  # shape = (2, 3, 4), axis = 1, broadcast shape = (2, 1, 4).
  # shape = (2, 3, 4), axis = 2, broadcast shape = (2, 3, 1).
  broadcast_shape = list(numerator.shape)
  broadcast_shape[axis] = 1

  # Get normalization term by summing along axis.
  denominator = numerator.sum(axis=axis).reshape(broadcast_shape)

  # Normalize to get softmax.
  return numerator / denominator
