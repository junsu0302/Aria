from Aria.core.Function import Function

class MeanSquaredError(Function):
  def forward(self, x0, x1):
    diff = x0 - x1
    return (diff ** 2).sum() / len(diff)
  
  def backward(self, gy):
    x0, x1 = self.inputs
    diff = x0 - x1
    gx0 = gy * diff * (2. / len(diff))
    gx1 = -gx0
    return gx0, gx1
  
def mean_squared_error(x0, x1):
  return MeanSquaredError()(x0, x1)