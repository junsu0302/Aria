import numpy as np

from Aria.core.Function import Function
from Aria.core.Math import logsumexp

from Aria.activation import softmax

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

class SoftmaxCrossEntropy(Function):
  def forward(self, x, t):
    N = x.shape[0]
    log_z = logsumexp(x, axis=1)
    log_p = x - log_z
    log_p = log_p[np.arange(N), t.ravel()]
    y = -log_p.sum() / np.float32(N)
    return y

  def backward(self, gy):
    x, t = self.inputs
    N, CLS_NUM = x.shape

    gy *= 1/N
    y = softmax(x)
    # convert to one-hot
    t_onehot = np.eye(CLS_NUM, dtype=t.dtype)[t.data]
    y = (y - t_onehot) * gy
    return y


def softmax_cross_entropy(x, t):
  return SoftmaxCrossEntropy()(x, t)