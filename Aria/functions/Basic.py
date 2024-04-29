import numpy as np
from Aria.core.Function import Function

class Sin(Function):
  def forward(self, x:np.ndarray) -> np.ndarray:
    return np.sin(x)
  
  def backward(self, gy:np.ndarray) -> np.ndarray:
    x, = self.inputs
    return gy * cos(x)
  
def sin(x:np.ndarray) -> np.ndarray:
  return Sin()(x)

class Cos(Function):
  def forward(self, x:np.ndarray) -> np.ndarray:
    return np.cos(x)
  
  def backward(self, gy:np.ndarray) -> np.ndarray:
    x, = self.inputs
    return gy * -sin(x)
  
def cos(x:np.ndarray):
  return Cos()(x)

class Tanh(Function):
  def forward(self, x:np.ndarray) -> np.ndarray:
    return np.tanh(x)
  
  def backward(self, gy:np.ndarray) -> np.ndarray:
    y = self.outputs[0]()
    return gy * (1 - y ** 2)
  
def tanh(x:np.ndarray) -> np.ndarray:
  return Tanh()(x)

