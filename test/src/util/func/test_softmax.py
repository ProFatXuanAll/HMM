import numpy as np
import pytest

import hmm.util.func


@pytest.fixture
def arr() -> np.ndarray:
  return np.arange(24).reshape(2, 3, 4)


@pytest.fixture(params=[0, 1, 2])
def axis(request) -> int:
  return request.param


def test_softmax_value(arr: np.ndarray, axis: int) -> None:
  """Must have correct values."""
  out_arr = hmm.util.func.softmax(arr=arr, axis=axis)
  assert out_arr.shape == arr.shape, 'must have correct shape.'
  assert np.all((0.0 <= out_arr) & (out_arr <= 1.0)), 'Values must be within the range [0, 1].'
  assert np.all(np.isclose(out_arr.sum(axis=axis), 1.0)), 'Values must sum up to 1.'
