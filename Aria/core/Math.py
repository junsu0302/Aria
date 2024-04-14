import numpy as np

from Aria.core.Function import Function
from Aria.core.Variable import Variable
from Aria.core.Utils import as_array

class Add(Function):
  def forward(self, x0, x1):
    return x0 + x1
  
  def backward(self, gy):
    return gy, gy
  
def add(x0, x1):
  x1 = as_array(x1)
  return Add()(x0, x1)

class Sub(Function):
  def forward(self, x0, x1):
    return x0 - x1
  
  def backward(self, gy):
    return gy, -gy
  
def sub(x0, x1):
  x1 = as_array(x1)
  return Sub()(x0, x1)

def rsub(x0, x1):
  x1 = as_array(x1)
  return Sub()(x1, x0)

class Mul(Function):
  def forward(self, x0, x1):
    return x0 * x1
  
  def backward(self, gy):
    x0, x1 = self.inputs[0].data, self.inputs[1].data
    return gy * x1, gy * x0

def mul(x0, x1):
  x1 = as_array(x1)
  return Mul()(x0, x1)

class Div(Function):
  def forward(self, x0, x1):
    return x0 / x1
  
  def backward(self, gy):
    x0, x1 = self.inputs[0].data, self.inputs[1].data
    gx0 = gy / x1
    gx1 = gy * (-x0 / x1 ** 2)
    return gx0, gx1
  
def div(x0, x1):
  x1 = as_array(x1)
  return Div()(x0, x1)

def rdiv(x0, x1):
  x1 = as_array(x1)
  return Div()(x1, x0)

class Neg(Function):
  def forward(self, x):
    return -x
  
  def backward(self, gy):
    return -gy
  
def neg(x):
  return Neg()(x)

class Pow(Function):
  def __init__(self, c):
    self.c = c

  def forward(self, x):
    return x ** self.c
  
  def backward(self, gy):
    x = self.inputs[0].data
    c = self.c
    return c * x ** (c-1) * gy
  
def pow(x, c):
  return Pow(c)(x)

class Square(Function):
  def forward(self, x):
    return x ** 2
  
  def backward(self, gy):
    x = self.inputs[0].data
    return 2 * x * gy
  
def square(x):
  return Square()(x);

class Exp(Function):
  def forward(self, x):
    return np.exp(x)
  
  def backward(self, gy):
    x = self.inputs[0].data
    return np.exp(x) * gy
  
def exp(x):
  return Exp()(x)

#! 수치 미분
def numerical_diff(f, x, eps=1e-4):
  x0 = Variable(x.data - eps)
  x1 = Variable(x.data + eps)
  y0 = f(x0)
  y1 = f(x1)
  return (y1.data - y0.data) / (2 * eps)
