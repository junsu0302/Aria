import numpy as np

from Aria.core.Function import Function
from Aria.core.Utils import as_varialbe

import Aria.functions.utils.Transform as Utils

class Sum(Function):
  def __init__(self, axis, keepdims):
    self.axis = axis # 계산 축
    self.keepdims = keepdims # 입출력의 차원수를 같게 유지할지 결정

  def forward(self, x):
    self.x_shape = x.shape # 형상
    return x.sum(axis=self.axis, keepdims=self.keepdims)
  
  def backward(self, gy):
    gy = Utils.reshape_sum_backward(gy, self.x_shape, self.axis, self.keepdims)
    return broadcast_to(gy, self.x_shape)
  
def sum(x, axis=None, keepdims=False):
  return Sum(axis, keepdims)(x)

class SumTo(Function):
  def __init__(self, shape):
    self.shape = shape # 형상

  def forward(self, x):
    self.x_shape = x.shape
    return Utils.sum_to(x, self.shape)
  
  def backward(self, gy):
    return broadcast_to(gy, self.x_shape)
  
def sum_to(x, shape):
  if x.shape == shape:
    return as_varialbe(x)
  
  return SumTo(shape)(x)

class MatMul(Function):
  def forward(self, x, W):
    return x.dot(W)
  
  def backward(self, gy):
    x, W = self.inputs
    gx, gW = matmul(gy, W.T), matmul(x.T, gy)
    return gx, gW
  
def matmul(x, W):
  return MatMul()(x, W)

class Reshape(Function):
  def __init__(self, shape):
    self.shape = shape # 형상

  def forward(self, x):
    self.x_shape = x.shape
    return x.reshape(self.shape)
  
  def backward(self, gy):
    return reshape(gy, self.x_shape)
  
def reshape(x, shape):
  if x.shape == shape:
    return as_varialbe(x)
  return Reshape(shape)(x)

class Transpose(Function):
  def forward(self, x):
    return np.transpose(x)
  
  def backward(self, gy):
    return transpose(gy)
  
def transpose(x):
  return Transpose()(x)

class BroadcastTo(Function):
  def __init__(self, shape):
    self.shape = shape # 형상

  def forward(self, x):
    self.x_shape = x.shape
    return np.broadcast_to(x, self.shape)
  
  def backward(self, gy):
    return sum_to(gy, self.x_shape)
  
def broadcast_to(x, shape):
  if x.shape == shape:
    return as_varialbe(x)
  return BroadcastTo(shape)(x)