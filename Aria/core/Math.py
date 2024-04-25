import numpy as np

import Aria

from Aria.core.Function import Function
from Aria.core.Variable import Variable
from Aria.core.Utils import as_array

class Add(Function):
  def forward(self, x0, x1):
    """덧셈 연산의 forward를 수행하는 메서드

    Args:
      x0 (numpy.ndarray): 첫 번째 입력 데이터
      x1 (numpy.ndarray): 두 번째 입력 데이터

    Returns:
      numpy.ndarray: 덧셈의 결과
    """
    self.x0_shape, self.x1_shape = x0.shape, x1.shape
    return x0 + x1
  
  def backward(self, gy):
    """덧셈 연산의 backward를 수행하는 메서드

    Args:
      gy (numpy.ndarray): 출력 쪽에서 전해지는 기울기

    Returns:
      Tuple[numpy.ndarray, numpy.ndarray]: 입력 데이터에 대한 기울기
    """
    gx0, gx1 = gy, gy
    if self.x0_shape != self.x1_shape:
      gx0 = Aria.functions.Tensor.sum_to(gx0, self.x0_shape)
      gx1 = Aria.functions.Tensor.sum_to(gx1, self.x1_shape)
    return gx0, gx1
  
def add(x0, x1):
  """두 변수의 덧셈 연산을 수행하는 함수

  Args:
    x0 (Union[float, int, numpy.ndarray]): 첫 번째 입력 변수
    x1 (Union[float, int, numpy.ndarray]): 두 번째 입력 변수

  Returns:
    Variable: 덧셈 연산의 결과를 담은 변수
  """
  x1 = as_array(x1)
  return Add()(x0, x1)

class Sub(Function):
  def forward(self, x0, x1):
    """뺄셈 연산의 forward를 수행하는 메서드

    Args:
      x0 (numpy.ndarray): 첫 번째 입력 데이터
      x1 (numpy.ndarray): 두 번째 입력 데이터

    Returns:
      numpy.ndarray: 뺄셈의 결과
    """
    self.x0_shape, self.x1_shape = x0.shape, x1.shape
    return x0 - x1
  
  def backward(self, gy):
    """뺄셈 연산의 backward를 수행하는 메서드

    Args:
      gy (numpy.ndarray): 출력 쪽에서 전해지는 기울기

    Returns:
      Tuple[numpy.ndarray, numpy.ndarray]: 입력 데이터에 대한 기울기
    """
    gx0, gx1 = gy, -gy
    if self.x0_shape != self.x1_shape:
      gx0 = Aria.functions.Tensor.sum_to(gx0, self.x0_shape)
      gx1 = Aria.functions.Tensor.sum_to(gx1, self.x1_shape)
    return gx0, gx1
  
def sub(x0, x1):
  """두 변수의 뺄셈 연산을 수행하는 함수

  Args:
    x0 (Union[float, int, numpy.ndarray]): 첫 번째 입력 변수
    x1 (Union[float, int, numpy.ndarray]): 두 번째 입력 변수

  Returns:
    Variable: 뺄셈 연산의 결과를 담은 변수
  """
  x1 = as_array(x1)
  return Sub()(x0, x1)

def rsub(x0, x1):
  """첫 번째 변수에서 두 번째 변수를 뺄셈하는 함수

  Args:
    x0 (Union[float, int, numpy.ndarray]): 첫 번째 입력 변수
    x1 (Union[float, int, numpy.ndarray]): 두 번째 입력 변수

  Returns:
    Variable: 뺄셈 연산의 결과를 담은 변수
  """
  x1 = as_array(x1)
  return Sub()(x1, x0)

class Mul(Function):
  def forward(self, x0, x1):
    """곱셈 연산의 forward를 수행하는 메서드

    Args:
      x0 (numpy.ndarray): 첫 번째 입력 데이터
      x1 (numpy.ndarray): 두 번째 입력 데이터

    Returns:
      numpy.ndarray: 곱셈의 결과
    """
    self.x0_shape, self.x1_shape = x0.shape, x1.shape
    return x0 * x1
  
  def backward(self, gy):
    """곱셈 연산의 backward를 수행하는 메서드

    Args:
      gy (numpy.ndarray): 출력 쪽에서 전해지는 기울기

    Returns:
      Tuple[numpy.ndarray, numpy.ndarray]: 입력 데이터에 대한 기울기
    """
    x0, x1 = self.inputs
    gx0, gx1 = gy * x1, gy * x0
    if self.x0_shape != self.x1_shape:
      gx0 = Aria.functions.Tensor.sum_to(gx0, self.x0_shape)
      gx1 = Aria.functions.Tensor.sum_to(gx1, self.x1_shape)
    return gx0, gx1

def mul(x0, x1):
  """두 변수의 곱셈 연산을 수행하는 함수

  Args:
    x0 (Union[float, int, numpy.ndarray]): 첫 번째 입력 변수
    x1 (Union[float, int, numpy.ndarray]): 두 번째 입력 변수

  Returns:
    Variable: 곱셈 연산의 결과를 담은 변수
  """
  x1 = as_array(x1)
  return Mul()(x0, x1)

class Div(Function):
  def forward(self, x0, x1):
    """나눗셈 연산의 forward를 수행하는 메서드

    Args:
      x0 (numpy.ndarray): 첫 번째 입력 데이터
      x1 (numpy.ndarray): 두 번째 입력 데이터

    Returns:
      numpy.ndarray: 나눗셈의 결과
    """
    self.x0_shape, self.x1_shape = x0.shape, x1.shape
    return x0 / x1
  
  def backward(self, gy):
    """나눗셈 연산의 backward를 수행하는 메서드

    Args:
      gy (numpy.ndarray): 출력 쪽에서 전해지는 기울기

    Returns:
      Tuple[numpy.ndarray, numpy.ndarray]: 입력 데이터에 대한 기울기
    """
    x0, x1 = self.inputs
    gx0, gx1 = gy / x1, gy * (-x0 / x1 ** 2)
    if self.x0_shape != self.x1_shape:
      gx0 = Aria.functions.Tensor.sum_to(gx0, self.x0_shape)
      gx1 = Aria.functions.Tensor.sum_to(gx1, self.x1_shape)
    return gx0, gx1
  
def div(x0, x1):
  """두 변수의 나눗셈 연산을 수행하는 함수

  Args:
    x0 (Union[float, int, numpy.ndarray]): 첫 번째 입력 변수
    x1 (Union[float, int, numpy.ndarray]): 두 번째 입력 변수

  Returns:
    Variable: 나눗셈 연산의 결과를 담은 변수
    """
  x1 = as_array(x1)
  return Div()(x0, x1)

def rdiv(x0, x1):
  """두 변수의 역수를 나누는 연산을 수행하는 함수

  Args:
    x0 (Union[float, int, numpy.ndarray]): 첫 번째 입력 변수
    x1 (Union[float, int, numpy.ndarray]): 두 번째 입력 변수

  Returns:
    Variable: 나눗셈 연산의 결과를 담은 변수
  """
  x1 = as_array(x1)
  return Div()(x1, x0)

class Neg(Function):
  def forward(self, x):
    """음수 연산의 forward를 수행하는 메서드

    Args:
      x (numpy.ndarray): 입력 데이터

    Returns:
      numpy.ndarray: 음수의 결과
    """
    return -x
  
  def backward(self, gy):
    """음수 연산의 backward를 수행하는 메서드

    Args:
      gy (numpy.ndarray): 출력 쪽에서 전해지는 기울기

    Returns:
      numpy.ndarray: 입력 데이터에 대한 기울기
    """
    return -gy
  
def neg(x):
  """변수의 음수 연산을 수행하는 함수

   Args:
     x (Union[float, int, numpy.ndarray]): 입력 변수

  Returns:
    Variable: 음수 연산의 결과를 담은 변수
  """
  return Neg()(x)

class Pow(Function):
  def __init__(self, c):
    self.c = c

  def forward(self, x):
    """거듭제곱 연산의 forward를 수행하는 메서드

    Args:
      x (numpy.ndarray): 입력 데이터

    Returns:
      numpy.ndarray: 거듭제곱의 결과
    """
    return x ** self.c
  
  def backward(self, gy):
    """거듭제곱 연산의 backward를 수행하는 메서드

    Args:
      gy (numpy.ndarray): 출력 쪽에서 전해지는 기울기

    Returns:
      numpy.ndarray: 입력 데이터에 대한 기울기
    """
    x, = self.inputs
    c = self.c
    return c * x ** (c-1) * gy
  
def pow(x, c):
  """변수의 거듭제곱 연산을 수행하는 함수

  Args:
    x (Union[float, int, numpy.ndarray]): 입력 변수
    c (float): 거듭제곱할 지수

  Returns:
    Variable: 거듭제곱 연산의 결과를 담은 변수
    """
  return Pow(c)(x)

class Square(Function):
  def forward(self, x):
    """입력값을 제곱하여 반환

    Args:
      x (numpy.ndarray): 입력 데이터

    Returns:
      numpy.ndarray: 입력값을 제곱한 결과
    """
    return x ** 2
  
  def backward(self, gy):
    """제곱 연산의 backward를 수행

    Args:
      gy (numpy.ndarray): 출력 쪽에서 전해지는 기울기

    Returns:
      numpy.ndarray: 입력 데이터에 대한 기울기
    """
    x, = self.inputs
    return 2 * x * gy
  
def square(x):
  """변수를 제곱하는 함수

  Args:
    x (Union[float, int, numpy.ndarray]): 입력 변수

  Returns:
    Variable: 제곱된 결과를 담은 변수
    """
  return Square()(x);

class Exp(Function):
  def forward(self, x):
    """입력값의 지수 함수값을 반환

    Args:
      x (numpy.ndarray): 입력 데이터

    Returns:
      numpy.ndarray: 입력값의 지수 함수값
    """
    return np.exp(x)
  
  def backward(self, gy):
    """지수 함수의 backward를 수행

    Args:
      gy (numpy.ndarray): 출력 쪽에서 전해지는 기울기

    Returns:
      numpy.ndarray: 입력 데이터에 대한 기울기
    """
    y = self.outputs[0]()
    return gy * y
  
def exp(x):
  """변수의 지수 함수값을 계산하는 함수

  Args:
    x (Union[float, int, numpy.ndarray]): 입력 변수

  Returns:
    Variable: 지수 함수값을 담은 변수
  """
  return Exp()(x)

def logsumexp(x, axis=1):
  """입력값의 로그 합 지수 함수를 계산합니다.

  Args:
    x (numpy.ndarray): 입력 데이터입니다.
    axis (int, optional): 합을 구할 축입니다. 기본값은 1입니다.

  Returns:
    numpy.ndarray: 로그 합 지수 함수값입니다.
  """
  m = x.max(axis=axis, keepdims=True)
  y = x - m
  np.exp(y, out=y)
  s = y.sum(axis=axis, keepdims=True)
  np.log(s, out=s)
  m += s
  return m

def numerical_diff(f, x, eps=1e-4):
  """수치 미분을 수행하여 기울기를 계산

  Args:
    f (Callable): 미분할 함수
    x (Variable): 미분을 수행할 변수
    eps (float, optional): 미분 계산에 사용할 작은 값(기본값은 1e-4)

  Returns:
    numpy.ndarray: 수치 미분을 통해 계산된 기울기
    """
  x0 = Variable(x.data - eps)
  x1 = Variable(x.data + eps)
  y0 = f(x0)
  y1 = f(x1)
  return (y1.data - y0.data) / (2 * eps)
