import inspect
from inspect import Parameter, Signature

import numpy as np

import src.util.func


def test_module_method() -> None:
  """Ensure module methods' signature."""
  assert hasattr(src.util.func, 'softmax')
  assert inspect.isfunction(src.util.func.softmax)
  assert inspect.signature(src.util.func.softmax) == Signature(
    parameters=[
      Parameter(
        annotation=np.ndarray,
        default=Parameter.empty,
        kind=Parameter.KEYWORD_ONLY,
        name='arr',
      ),
      Parameter(
        annotation=int,
        default=0,
        kind=Parameter.KEYWORD_ONLY,
        name='axis',
      ),
    ],
    return_annotation=np.ndarray,
  )
