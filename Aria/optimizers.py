import numpy as np
import math

from Aria.core.Optimizer import Optimizer

class SGD(Optimizer):
  """
  확률적 경사 하강법

  - init(lr)
  - update_one(param): 단일 파라미터 업데이트
  """
  def __init__(self, lr:float=0.01) -> None:
    """확률적 경사 하강법

    Args:
      lr (float, optional): 학습률(기본값은 0.01)
    """
    super().__init__()
    self.lr = lr # 적용률

  def update_one(self, param):
    """단일 파라미터 업데이트"""
    param.data -= self.lr * param.grad.data # 매개변수 갱신

class MomentumSGD(Optimizer):
  """
  모멘텀 SGD

  - init(lr, mometum)
  - update_one(param): 단일 파라미터 업데이트
  """
  def __init__(self, lr:float=0.01, momentum:float=0.9) -> None:
    """모멘텀 SGD

    Args:
      lr (float, optional): 학습률(기본값은 0.01)
      momentum (float, optional): 적용률(기본값은 0.9)
    """
    super().__init__()
    self.lr = lr # 학습률
    self.momentum = momentum # 적용률
    self.vs = {} # 속도

  def update_one(self, param):
    """단일 파라미터 업데이트"""
    v_key = id(param)
    if v_key not in self.vs:
      self.vs[v_key] = np.zeros_like(param.data)

    v = self.vs[v_key]
    v *= self.momentum
    v -= self.lr * param.grad.data
    param.data += v

class Adam(Optimizer):
  """
  Adam

  - init(alpha, beta1, beta2, eps)
  - update(*args, **kwargs): 파라미터 업데이트
  - update_one(param): 단일 파라미터 업데이트
  """
  def __init__(self, alpha:float=0.001, beta1:float=0.9, beta2:float=0.999, eps:float=1e-8) -> None:
    """Adam

    Args:
      alpha (float, optional): 초기 학습률(기본값은 0.001)
      beta1 (float, optional): 첫 번째 모멘트 추정의 감쇠율(기본값은 0.9)
      beta2 (float, optional): 두 번째 모멘트 추정의 감쇠율(기본값은 0.999)
      eps (float, optional): 수치 안정성을 위해 분모에 추가되는 노이즈(기본값은 1e-8)
    """
    super().__init__()
    self.t = 0 # 시간
    self.alpha = alpha # 초기 학습률
    self.beta1 = beta1 # 첫 번째 모멘트 추정의 감쇠율
    self.beta2 = beta2 # 두 번째 모멘트 추정의 감쇠율
    self.eps = eps # 수치 안정성을 위한 노이즈
    self.ms = {} # 기울기의 지수적으로 감소하는 이동 평균을 유지하는 딕셔너리
    self.vs = {} # 제곱된 기울기의 지수적으로 감소하는 이동 평균을 유지하는 딕셔너리

  @property
  def lr(self) -> float:
    """학습률 계산"""
    fix1 = 1. - math.pow(self.beta1, self.t)
    fix2 = 1. - math.pow(self.beta2, self.t)
    return self.alpha * math.sqrt(fix2) / fix1

  def update(self, *args, **kwargs):
    """파라미터 업데이트"""
    self.t += 1
    super().update(*args, **kwargs)

  def update_one(self, param):
    """단일 파라미터 업데이트"""
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