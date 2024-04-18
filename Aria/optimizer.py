import numpy as np

from Aria.core.Optimizer import Optimizer

class SGD(Optimizer):
  def __init__(self, lr=0.01):
    super().__init__()
    self.lr = lr # 적용률

  def update_one(self, param):
    param.data -= self.lr * param.grad.data # 매개변수 갱신

class MomentumSGD(Optimizer):
  def __init__(self, lr=0.01, momentum=0.9):
    super().__init__()
    self.lr = lr # 학습률
    self.momentum = momentum # 적용률
    self.vs = {} # 속도

  def update_one(self, param):
    v_key = id(param)
    if v_key not in self.vs:
      self.vs[v_key] = np.zeros_like(param.data)

    v = self.vs[v_key]
    v *= self.momentum
    v -= self.lr * param.grad.data
    param.data += v