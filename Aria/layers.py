import numpy as np

from Aria.core.Layer import Layer
from Aria.core.Parameter import Parameter

from Aria.functions.Convolutional import conv2d
from Aria.functions.Tensor import linear

from Aria.functions.utils.Convolutional import pair

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

    return linear(x, self.W, self.b)
  
class Conv2d(Layer):
  def __init__(self, out_channels, kernel_size, stride=1, pad=0, nobias=False, dtype=np.float32, in_channels=None):
    super().__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.kernel_size = kernel_size
    self.stride = stride
    self.pad = pad
    self.dtype = dtype

    self.W = Parameter(None, name='W')
    if in_channels is not None:
      self._init_W()

    if nobias:
      self.b = None
    else:
      self.b = Parameter(np.zeros(out_channels, dtype=dtype), name='b')
    
  def _init_W(self, xp):
    C, OC = self.in_channels, self.out_channels
    KH, KW = pair(self.kernel_size)
    scale = np.sqrt(1 / (C * KH * KW))
    W_data = xp.random.randn(OC, C, KH, KW).astype(self.dtype) * scale
    self.W.data = W_data

  def forward(self, x):
    if self.W.data is None:
      self.in_channels = x.shape[1]
      self._init_W(x)

    return conv2d(x, self.W, self.b, self.stride, self.pad)