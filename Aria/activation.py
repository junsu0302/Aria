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