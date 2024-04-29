import numpy as np

from Aria.core.Function import Function
from Aria.core.Utils import as_variable

import Aria.functions.utils.Transform as Utils

class Reshape(Function):
  def __init__(self, shape):
    """다차원 배열의 형태를 변환하는 함수

    Args:
      shape (Union[int, Tuple[int, ...]]): 형태를 변환할 다차원 배열의 차원
    """
    self.shape = shape # 형상

  def forward(self, x):
    """forward를 수행하는 메서드

    Args:
      x (numpy.ndarray): 입력 다차원 배열

    Returns:
      numpy.ndarray : 입력 다차원 배열의 형태가 변환된 결과를 나타내는 배열
    """
    self.x_shape = x.shape
    return x.reshape(self.shape)
  
  def backward(self, gy):
    """backward를 수행하는 메서드

    Args:
      gy (numpy.ndarray): 출력의 기울기

    Returns:
      numpy.ndarray: 입력의 기울기를 나타내는 배열
    """
    return reshape(gy, self.x_shape)
  
def reshape(x, shape):
  """다차원 배열의 형태를 변환하는 함수

  Args:
    x (numpy.ndarray): 형태를 변환할 다차원 배열
    shape (Union[int, Tuple[int, ...]]): 형태를 변환할 다차원 배열의 차원

  Returns:
    numpy.ndarray : 형태가 변환된 다차원 배열을 나타내는 배열
  """
  if x.shape == shape:
    return as_variable(x)
  return Reshape(shape)(x)

class Transpose(Function):
  def __init__(self, axes=None):
    """다차원 배열의 축을 변경하는 함수

    Args:
      axes (Optional[Union[int, Tuple[int, ...]]]): 변경할 축을 나타내는 값(기본값은 None)
    """
    self.axes = axes

  def forward(self, x):
    """forward를 수행하는 메서드

    Args:
      x (numpy.ndarray): 입력 다차원 배열

    Returns:
      numpy.ndarray: 입력 다차원 배열의 축이 변경된 결과를 나타내는 배열
    """
    return x.transpose(self.axes)
  
  def backward(self, gy):
    """backward를 수행하는 메서드

    Args:
      gy (numpy.ndarray): 출력의 기울기

    Returns:
      numpy.ndarray: 입력의 기울기를 나타내는 배열
    """
    if self.axes is None:
      return transpose(gy)
    
    axes_len = len(self.axes)
    inv_axes = tuple(np.argsort([ax % axes_len for ax in self.axes]))
    return transpose(gy, axes=inv_axes)
  
def transpose(x, axes=None):
  """다차원 배열의 축을 변경하는 함수

  Args:
    x (numpy.ndarray): 변경할 다차원 배열
    axes (Optional[Union[int, Tuple[int, ...]]]): 변경할 축을 나타내는 값(기본값은 None)

  Returns:
    numpy.ndarray: _description_
  """
  return Transpose(axes)(x)

class Sum(Function):
  def __init__(self, axis, keepdims):
    """다차원 배열의 합을 계산하는 함수

    Args:
      axis (Optional[int | Tuple[int]]): 합을 계산할 축을 나하태는 정수
      keepdims (Optional[bool]): 출력에서 입력 차원 형태를 유지할지 여부를 나타내는 플래그
    """
    self.axis = axis # 계산 축
    self.keepdims = keepdims # 입출력 차원 수 유지 모드

  def forward(self, x):
    """forward를 수행하는 메서드

    Args:
      x (numpy.ndarray): 입력 다차원 배열

    Returns:
      numpy.ndarray: 입력 다차원 배열의 합이 계산된 결과를 나타내는 배열
    """
    self.x_shape = x.shape
    return x.sum(axis=self.axis, keepdims=self.keepdims)
  
  def backward(self, gy):
    """backward를 수행하는 메서드

    Args:
      gy (numpy.ndarray): 출력의 기울기

    Returns:
      numpy.ndarray: 입력의 기울기를 나타내는 배열
    """
    gy = Utils.reshape_sum_backward(gy, self.x_shape, self.axis, self.keepdims)
    return broadcast_to(gy, self.x_shape)
  
def sum(x, axis=None, keepdims=False):
  """다차원 배열의 합을 계산하는 함수

    Args:
      x (numpy.ndarray): 합을 계산할 다차원 배열
      axis (Optional[int | Tuple[int]]): 합을 계산할 축을 나하태는 정수(기본값은 None)
      keepdims (Optional[bool]): 출력에서 입력 차원 형태를 유지할지 여부를 나타내는 플래그(기본값은 False)
    
    Returns:
      numpy.ndarray: 합이 계산된 다차원 배열을 나타내는 배열
    """
  return Sum(axis, keepdims)(x)

class BroadcastTo(Function):
  def __init__(self, shape):
    """지정된 형태로 브로드캐스팅하는 클래스

    Args:
      shape (Tuple[int]): 브로드캐스팅할 형태
    """
    self.shape = shape

  def forward(self, x):
    """forward를 수행하는 메서드

    Args:
      x (numpy.ndarray): 입력 배열

    Returns:
      numpy.ndarray: 브로드캐스팅된 배열
    """
    self.x_shape = x.shape
    return np.broadcast_to(x, self.shape)
  
  def backward(self, gy):
    """backward를 수행하는 메서드

    Args:
      gy (numpy.ndarray): 출력의 기울기

    Returns:
      numpy.ndarray: 입력의 기울기를 나타내는 배열
    """
    return sum_to(gy, self.x_shape)
  
def broadcast_to(x, shape):
  """입력 배열을 지정된 모양으로 브로드캐스팅

  Args:
    x (numpy.ndarray): 입력 배열
    shape (Tuple[int]): 브로드캐스팅할 모양

  Returns:
    numpy.ndarray: 브로드캐스팅된 배열
  """
  if x.shape == shape:
    return as_variable(x)
  return BroadcastTo(shape)(x)

class SumTo(Function):
  def __init__(self, shape):
    """지정된 모양으로 변환하는 클래스

    Args:
      shape (Tuple[int]): 더할 모양
    """
    self.shape = shape

  def forward(self, x):
    """forward를 수행하는 메서드

    Args:
      x (numpy.ndarray): 입력 배열

    Returns:
      numpy.ndarray: 지정된 모양으로 더해진 배열
    """
    self.x_shape = x.shape
    return Utils.sum_to(x, self.shape)
  
  def backward(self, gy):
    """backward를 수행하는 메서드

    Args:
      gy (numpy.ndarray): 출력의 기울기

    Returns:
      numpy.ndarray: 입력의 기울기를 나타내는 배열
    """
    return broadcast_to(gy, self.x_shape)
  
def sum_to(x, shape):
  """입력 배열을 지정된 모양으로 변환하는 함수

  Args:
    x (numpy.ndarray): 입력 배열
    shape (Tuple[int]): 변환할 모양

  Returns:
    numpy.ndarray: 지정된 모양으로 변환된 배열
  """
  if x.shape == shape:
    return as_variable(x)
  return SumTo(shape)(x)

class MatMul(Function):
  def forward(self, x, W):
    """forward를 수행하는 메서드

    Args:
        x (numpy.ndarray): 입력 배열
        W (numpy.ndarray): 가중치 행렬

    Returns:
        numpy.ndarray: 행렬 곱 결과
    """
    return x.dot(W)
  
  def backward(self, gy):
    """backward를 수행하는 메서드

    Args:
      gy (numpy.ndarray): 출력의 기울기

    Returns:
      Tuple[numpy.ndarray, numpy.ndarray]: 입력의 기울기를 나타내는 배열
    """
    x, W = self.inputs
    gx = matmul(gy, W.T)
    gW = matmul(x.T, gy)
    return gx, gW
  
def matmul(x, W):
  """행렬 곱을 계산하는 함수

  Args:
    x (numpy.ndarray): 입력 배열
    W (numpy.ndarray): 가중치 행렬

  Returns:
    numpy.ndarray: 행렬 곱 결과
  """
  return MatMul()(x, W)

class Max(Function):
  def __init__(self, axis=None, keepdims=False):
    """최댓값을 계산하는 클래스

    Args:
      axis (Optional[int]): 최댓값을 계산할 축(기본값은 None)
      keepdims (Optional[bool]): 차원 수를 유지할지 여부를 결정하는 플래그.(기본값은 False)
    """
    self.axis = axis
    self.keepdims = keepdims

  def forward(self, x):
    """forward를 수행하는 메서드

    Args:
      x (numpy.ndarray): 입력 배열

    Returns:
      numpy.ndarray: 최댓값
    """
    y = x.max(axis=self.axis, keepdims=self.keepdims)
    return y

  def backward(self, gy):
    """backward를 수행하는 메서드

    Args:
      gy (numpy.ndarray): 출력의 기울기

    Returns:
      numpy.ndarray: 입력의 기울기를 나타내는 배열
    """
    x = self.inputs[0]
    y = self.outputs[0]()  # weakref

    shape = Utils.max_backward_shape(x, self.axis)
    gy = reshape(gy, shape)
    y = reshape(y, shape)
    cond = (x.data == y.data)
    gy = broadcast_to(gy, cond.shape)
    return gy * cond

def max(x, axis=None, keepdims=False):
  """최댓값을 계산하는 함수

  Args:
    x (numpy.ndarray): 입력 배열
    axis (Optional[int]): 최댓값을 계산할 축(기본값은 None)
    keepdims (Optional[bool]): 차원 수를 유지할지 결정하는 플래그(기본값은 False)

  Returns:
    numpy.ndarray: 최댓값
  """
  return Max(axis, keepdims)(x)

class Min(Max):
  def forward(self, x):
    """forward를 수행하는 메서드

    Args:
      x (numpy.ndarray): 입력 배열

    Returns:
      numpy.ndarray: 최솟값
    """
    y = x.min(axis=self.axis, keepdims=self.keepdims)
    return y

def min(x, axis=None, keepdims=False):
  """최솟값을 계산하는 함수

  Args:
    x (numpy.ndarray): 입력 배열
    axis (Optional[int], optional): 최솟값을 계산할 축(기본값은 None)
    keepdims (Optional[bool], optional): 차원 수를 유지할지 결정하는 플래그(기본값은 False)

  Returns:
    numpy.ndarray: 최솟값
  """
  return Min(axis, keepdims)(x)

class Linear(Function):
  """선형 변환을 수행하는 클래스"""
  def forward(self, x:np.ndarray, W:np.ndarray, b:np.ndarray) -> np.ndarray:
    """forward를 수행하는 메서드

    Args:
      x (numpy.ndarray): 입력 배열
      W (numpy.ndarray): 가중치 배열
      b (Optional[numpy.ndarray]): 편향 벡터

    Returns:
      numpy.ndarray: 선형 변환 결과
    """
    y = x.dot(W)
    if b is not None:
      y += b
    return y

  def backward(self, gy:np.ndarray):
    """backward를 수행하는 메서드

    Args:
      gy (numpy.ndarray): 출력의 기울기

    Returns:
      Tuple[numpy.ndarray, numpy.ndarray, Optional[numpy.ndarray]]: 입력의 기울기를 나타내는 배열
    """
    x, W, b = self.inputs
    gb = None if b.data is None else sum_to(gy, b.shape)
    gx = matmul(gy, W.T)
    gW = matmul(x.T, gy)
    return gx, gW, gb


def linear(x:np.ndarray, W:np.ndarray, b:np.ndarray=None):
  """선형 변환을 수행하는 함수

  Args:
    x (numpy.ndarray): 입력 배열
    W (numpy.ndarray): 가중치 행렬
    b (Optional[numpy.ndarray]): 편향 벡터(기본값은 None)

  Returns:
    numpy.ndarray: 선형 변환 결과
  """
  return Linear()(x, W, b)
