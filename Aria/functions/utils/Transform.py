from typing import Union
import numpy as np

def sum_to(x:np.ndarray, shape: tuple) -> np.ndarray:
  """shape에 맞게 입력 배열 합산

  Args:
    x (np.ndarray): 합산할 배열
    shape (tuple): 결과 배열의 모양

  Returns:
    np.ndarray: shape에 맞게 합산된 배열
  """
  ndim = len(shape)
  lead = x.ndim - ndim
  lead_axis = tuple(range(lead))

  axis = tuple([i + lead for i, sx in enumerate(shape) if sx == 1])
  y = x.sum(lead_axis + axis, keepdims=True)
  if lead > 0:
    y = y.squeeze(lead_axis)
  return y


def reshape_sum_backward(gy:np.ndarray, x_shape:tuple, axis:Union[int,tuple,None], keepdims:bool) -> np.ndarray:
  """합산의 역전파를 위한 reshape 작업 수행

  Args:
    gy (np.ndarray): 출력 기울기
    x_shape (tuple): 입력 배열의 모양
    axis (int | tuple | None): 합산할 축
    keepdims (bool): 축을 유지할지 결정하는 플래그

  Returns:
    np.ndarray: 재구성된 출력 기울기
  """
  ndim = len(x_shape)
  tupled_axis = axis
  if axis is None:
    tupled_axis = None
  elif not isinstance(axis, tuple):
    tupled_axis = (axis,)

  if not (ndim == 0 or tupled_axis is None or keepdims):
    actual_axis = [a if a >= 0 else a + ndim for a in tupled_axis]
    shape = list(gy.shape)
    for a in sorted(actual_axis):
      shape.insert(a, 1)
  else:
    shape = gy.shape

  gy = gy.reshape(shape)  # reshape
  return gy

def max_backward_shape(x:np.ndarray, axis:Union[int,tuple,None]) -> list:
  """최댓값의 역전파를 위한 출력 모양 계산

  Args:
    x (np.ndarray): 입력 배열
    axis (int | tuple | None]): 최댓값을 취할 축

  Returns:
    list: 최댓값의 역전파를 위한 출력 모양
  """
  if axis is None:
    axis = range(x.ndim)
  elif isinstance(axis, int):
    axis = (axis,)
  else:
    axis = axis

  shape = [s if ax not in axis else 1 for ax, s in enumerate(x.shape)]
  return shape