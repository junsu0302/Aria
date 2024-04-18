import numpy as np

import Aria.functions.Tensor as F

from Aria.core.Layer import Layer
from Aria.core.Parameter import Parameter

class Linear(Layer):
  def __init__(self, out_size, nobias=False, dtype=np.float32, in_size=None):
    super().__init__()
    self.in_size = in_size # 입력 크기
    self.out_size = out_size # 출력 크기
    self.dtype = dtype # 타입
    # in_size가 지정되지 않았다면 나중으로 연기
    self.W = Parameter(None, name='W')
    if self.in_size is not None:
      self._init_W()

    if nobias:
      self.b = None
    else:
      self.b = Parameter(np.zeros(out_size, dtype=dtype), name='b')

  def _init_W(self):
    I, O = self.in_size, self.out_size
    self.W.data =  np.random.randn(I, O).astype(self.dtype) * np.sqrt(1 / I)
  
  
  def forward(self, x):
    # 데이터를 보내는 시점에서 가중치 초기화
    if self.W.data is None:
      self.in_size = x.shape[1]
      self._init_W()

    return F.linear(x, self.W, self.b)