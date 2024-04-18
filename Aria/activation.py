import numpy as np

from Aria.core.Function import Function

class Sigmoid(Function):
  def forward(self, x):
    return 1 / (1 + np.exp(-x))
  
  def backward(self, gy):
    y = self.outputs[0]()
    return gy * y * (1 - y)

def sigmoid(x):
  return Sigmoid()(x)

class Softmax(Function):
  def __init__(self, axis=1):
    self.axis = axis

  def forward(self, x):
    return super().forward(x)
  
class Softmax(Function):
  def __init__(self, axis=1):
    self.axis = axis

  def forward(self, x):
    y = x - x.max(axis=self.axis, keepdims=True)
    y = np.exp(y)
    y /= y.sum(axis=self.axis, keepdims=True)
    return y

  def backward(self, gy):
    y = self.outputs[0]()
    gx = y * gy
    sumdx = gx.sum(axis=self.axis, keepdims=True)
    gx -= y * sumdx
    return gx


def softmax(x, axis=1):
  return Softmax(axis)(x)
