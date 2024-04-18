import numpy as np

from Aria.core.Function import Function
from Aria.core.Utils import as_variable

import Aria.functions.utils.Transform as Utils

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
    return as_variable(x)
  return Reshape(shape)(x)

class Transpose(Function):
  def forward(self, x):
    return np.transpose(x)
  
  def backward(self, gy):
    return transpose(gy)
  
def transpose(x):
  return Transpose()(x)

class Sum(Function):
  def __init__(self, axis, keepdims):
    self.axis = axis # 계산 축
    self.keepdims = keepdims # 입출력 차원 수 유지 모드

  def forward(self, x):
    self.x_shape = x.shape
    return x.sum(axis=self.axis, keepdims=self.keepdims)
  
  def backward(self, gy):
    gy = Utils.reshape_sum_backward(gy, self.x_shape, self.axis, self.keepdims)
    return broadcast_to(gy, self.x_shape)
  
def sum(x, axis=None, keepdims=False):
  return Sum(axis, keepdims)(x)

class BroadcastTo(Function):
  def __init__(self, shape):
    self.shape = shape

  def forward(self, x):
    self.x_shape = x.shape
    return np.broadcast_to(x, self.shape)
  
  def backward(self, gy):
    return sum_to(gy, self.x_shape)
  
def broadcast_to(x, shape):
  if x.shape == shape:
    return as_variable(x)
  return BroadcastTo(shape)(x)

class SumTo(Function):
  def __init__(self, shape):
    self.shape = shape

  def forward(self, x):
    self.x_shape = x.shape
    return Utils.sum_to(x, self.shape)
  
  def backward(self, gy):
    return broadcast_to(gy, self.x_shape)
  
def sum_to(x, shape):
  if x.shape == shape:
    return as_variable(x)
  return SumTo(shape)(x)

class MatMul(Function):
  def forward(self, x, W):
    return x.dot(W)
  
  def backward(self, gy):
    x, W = self.inputs
    gx = matmul(gy, W.T)
    gW = matmul(x.T, gy)
    return gx, gW
  
def matmul(x, W):
  return MatMul()(x, W)

class Max(Function):
  def __init__(self, axis=None, keepdims=False):
    self.axis = axis
    self.keepdims = keepdims

  def forward(self, x):
    y = x.max(axis=self.axis, keepdims=self.keepdims)
    return y

  def backward(self, gy):
    x = self.inputs[0]
    y = self.outputs[0]()  # weakref

    shape = Utils.max_backward_shape(x, self.axis)
    gy = reshape(gy, shape)
    y = reshape(y, shape)
    cond = (x.data == y.data)
    gy = broadcast_to(gy, cond.shape)
    return gy * cond

def max(x, axis=None, keepdims=False):
  return Max(axis, keepdims)(x)

class Min(Max):
  def forward(self, x):
    y = x.min(axis=self.axis, keepdims=self.keepdims)
    return y

def min(x, axis=None, keepdims=False):
  return Min(axis, keepdims)(x)

class Linear(Function):
  def forward(self, x, W, b):
    y = x.dot(W)
    if b is not None:
      y += b
    return y

  def backward(self, gy):
    x, W, b = self.inputs
    gb = None if b.data is None else sum_to(gy, b.shape)
    gx = matmul(gy, W.T)
    gW = matmul(x.T, gy)
    return gx, gW, gb


def linear(x, W, b=None):
  return Linear()(x, W, b)
