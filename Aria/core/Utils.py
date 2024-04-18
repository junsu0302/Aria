import numpy as np

from Aria.core.Function import Function

def as_array(x):
  import numpy as np

  if np.isscalar(x):
    return np.array(x)
  return x

def as_varialbe(obj):
  from Aria.core.Variable import Variable
  
  if isinstance(obj, Variable):
    return obj
  return Variable(obj)

class GetItem(Function):
  def __init__(self, slices):
    self.slices = slices

  def forward(self, x):
    return x[self.slices]
  
  def backward(self, gy):
    x, = self.inputs
    f = GetItemGrad(self.slices, x.shape)

class GetItemGrad(Function):
  def __init__(self, slices, in_shape):
    self.slices = slices
    self.in_shape = in_shape

  def forward(self, gy):
    gx = np.zeros(self.in_shape)
    np.add.at(gx, self.slices, gy)
    return gx
  
  def backward(self, ggx):
    return get_item(ggx, self.slices)
  
def get_item(x, slices):
  f = GetItem(slices)
  return f(x)
