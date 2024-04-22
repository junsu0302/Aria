import numpy as np
import math

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

class Adam(Optimizer):
  def __init__(self, alpha=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
    super().__init__()
    self.t = 0
    self.alpha = alpha
    self.beta1 = beta1
    self.beta2 = beta2
    self.eps = eps
    self.ms = {}
    self.vs = {}

  @property
  def lr(self):
    fix1 = 1. - math.pow(self.beta1, self.t)
    fix2 = 1. - math.pow(self.beta2, self.t)
    return self.alpha * math.sqrt(fix2) / fix1

  def update(self, *args, **kwargs):
    self.t += 1
    super().update(*args, **kwargs)

  def update_one(self, param):
    key = id(param)
    if key not in self.ms:
      self.ms[key] = np.zeros_like(param.data)
      self.vs[key] = np.zeros_like(param.data)

    m, v = self.ms[key], self.vs[key]
    beta1, beta2, eps = self.beta1, self.beta2, self.eps
    grad = param.grad.data

    m += (1 - beta1) * (grad - m)
    v += (1 - beta2) * (grad * grad - v)
    param.data -= self.lr * m / (np.sqrt(v) + eps)