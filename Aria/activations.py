import numpy as np

from Aria.core.Function import Function

class Sigmoid(Function):
  """
  Sigmoid 함수를 구현
  
  - forward(x): Sigmoid 함수의 순전파 수행
  - backward(gy): Sigmoid 함수의 역전파 수행
  """
  def forward(self, x:np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))
  
  def backward(self, gy:np.ndarray) -> np.ndarray:
    y = self.outputs[0]()
    return gy * y * (1 - y)

def sigmoid(x:np.ndarray) -> np.ndarray:
  """주어진 배열에 대한 Sigmoid 함수를 적용

  Args:
    x (numpy.ndarray): 입력 배열

  Returns:
    numpy.ndarray: 시그모이드 함수의 출력
  """
  return Sigmoid()(x)

class ReLU(Function):
  """
  ReLU 함수를 구현
  
  - forward(x): ReLU 함수의 순전파 수행
  - backward(gy): ReLU 함수의 역전파 수행
  """
  def forward(self, x:np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)

  def backward(self, gy:np.ndarray) -> np.ndarray:
    x, = self.inputs
    mask = x.data > 0
    return gy * mask

def relu(x:np.ndarray) -> np.ndarray:
  """주어진 배열에 대한 ReLU 함수를 적용

  Args:
    x (np.ndarray): 입력 배열

  Returns:
    np.ndarray: ReLU 함수의 출력
  """
  return ReLU()(x)
  
class Softmax(Function):
  """
  Softmax 함수를 구현
  
  - forward(x): Softmax 함수의 순전파 수행
  - backward(gy): Softmax 함수의 역전파 수행
  """  
  def __init__(self, axis:int=1):
    self.axis = axis

  def forward(self, x:np.ndarray) -> np.ndarray:
    y = x - x.max(axis=self.axis, keepdims=True)
    y = np.exp(y)
    y /= y.sum(axis=self.axis, keepdims=True)
    return y

  def backward(self, gy:np.ndarray) -> np.ndarray:
    y = self.outputs[0]()
    gx = y * gy
    sumdx = gx.sum(axis=self.axis, keepdims=True)
    gx -= y * sumdx
    return gx

def softmax(x:np.ndarray, axis:int=1) -> np.ndarray:
  """주어진 배열에 대한 Softmax 함수를 적용

  Args:
    x (np.ndarray): 입력 배열
    axis (int, optional): 연산할 출(기본값은 1)

  Returns:
    np.ndarray: Softmax 함수의 출력
  """
  return Softmax(axis)(x)
